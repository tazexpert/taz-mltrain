"""Bar-by-bar emulation of the SSS (Smart Structure Scalper) entry + exit logic of
`TAZ Expert Strategy-gemini.mq4` v7.22 on M5 OHLC data.

Index convention: `t` is the bar that has just CLOSED (MT4 shift 1). The EA acts on the
first tick of bar t+1, so orders fill at open[t+1] (Ask = Bid + spread for BUY).
CSV prices are Bid.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import EAConfig
from .data import resample
from .indicators import atr, ema, macd_main, filtered_avg_body


# ----------------------------------------------------------------------------------------
# Pre-computation
# ----------------------------------------------------------------------------------------
@dataclass
class Market:
    df: pd.DataFrame
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    spread: np.ndarray        # price units, per bar
    times: np.ndarray         # bar open times (datetime64)
    dec_time: pd.DatetimeIndex  # decision time = open of bar t+1
    ema: np.ndarray
    atr: np.ndarray
    avg_body: np.ndarray
    sig: np.ndarray           # SSS_GetSignalAt for each bar
    sw_hi: np.ndarray
    sw_lo: np.ndarray
    last_sig: np.ndarray      # index of the most recent bar <= i with sig != 0 (-1 if none)
    box_ok: np.ndarray        # M15 sideway box active at decision time (M15 shift 2)
    box_hi: np.ndarray
    box_lo: np.ndarray
    vol_ok: np.ndarray        # Guard 3 (H1 ATR circuit breaker) at decision time
    hhmm: np.ndarray          # decision time HHMM
    day: np.ndarray           # decision day (int)


def m15_sideway_boxes(m15: pd.DataFrame, cfg: EAConfig) -> pd.DataFrame:
    """IsMACDSidewaysAt() evaluated with every M15 bar as `baseShift`. Uses only that bar and older."""
    lb, base = cfg.sw_lookback, cfg.macd_baseline_bars
    c = m15["close"]
    macd = macd_main(c, cfg.macd_fast, cfg.macd_slow)
    rec_rng = macd.rolling(lb).max() - macd.rolling(lb).min()
    base_rng = macd.rolling(base).max() - macd.rolling(base).min()
    a = atr(m15, cfg.sw_atr_period)
    ma_dist = (ema(c, cfg.ma_fast) - ema(c, cfg.ma_slow)).abs()
    hi = m15["high"].rolling(lb).max()
    lo = m15["low"].rolling(lb).min()
    body_avg = (m15["close"] - m15["open"]).abs().rolling(lb).mean()

    d = np.sign(m15["close"] - m15["open"]).to_numpy()
    streak = np.zeros(len(d))
    run, last = 0, 0
    for i, x in enumerate(d):  # streak of same colour, dojis neither break nor extend (as in EA)
        if x != 0 and x == last:
            run += 1
        else:
            run = 1 if x != 0 else 0
        if x != 0:
            last = x
        streak[i] = run
    max_streak = pd.Series(streak, index=m15.index).rolling(lb).max()
    # NOTE: the EA restarts its streak counter at the window start; a rolling max of a global
    # running streak can over-count by carrying a streak in from before the window. Fine for
    # max_same_color_streak=5 vs lookback=8 except edge cases.

    ok = (base_rng > 0) & ((rec_rng / base_rng) <= cfg.macd_flat_ratio)
    ok &= (a > 0) & (ma_dist <= a * cfg.ma_dist_atr_ratio)
    ok &= (hi - lo) <= a * cfg.range_atr_ratio
    ok &= (body_avg / a <= cfg.body_atr_ratio) & (max_streak <= cfg.max_same_color_streak)
    ok &= np.arange(len(m15)) >= base + cfg.macd_slow  # warm-up
    return pd.DataFrame({"ok": ok.fillna(False), "hi": hi, "lo": lo}, index=m15.index)


def prepare(df: pd.DataFrame, cfg: EAConfig) -> Market:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    spread = (df["spread"].to_numpy(float) * cfg.point) if "spread" in df.columns else np.full(n, cfg.spread)

    e = ema(df["close"], cfg.ema_slope).to_numpy()
    a = atr(df, 14).to_numpy()
    avg_body = filtered_avg_body(o, c, 20)

    # --- SSS_GetSignalAt(shift) for every bar k ---
    sb = cfg.swing_bars
    sw_hi = pd.Series(h).shift(1).rolling(sb).max().to_numpy()
    sw_lo = pd.Series(l).shift(1).rolling(sb).min().to_numpy()
    slope = e - np.roll(e, 2)
    slope[:2] = 0
    body = np.abs(c - o)
    body_prev = np.roll(body, 1)
    body_prev[0] = np.inf
    engulf = body >= body_prev * cfg.engulf_ratio
    pt10 = cfg.point * 10.0  # EA quirk: ATR gate uses Point*10, every other distance uses Point
    atr_ok = (a >= cfg.atr_min_pts * pt10) & (a <= cfg.atr_max_pts * pt10)
    buy = atr_ok & (slope > 0) & (c > sw_hi) & (c > o) & engulf
    sell = atr_ok & (slope < 0) & (c < sw_lo) & (c < o) & engulf
    sig = np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)
    sig[: max(sb + 2, 30)] = 0

    idx = np.where(sig != 0, np.arange(n), -1)
    last_sig = np.maximum.accumulate(idx)

    # --- decision time and higher-timeframe context ---
    times = df.index.values
    step = pd.Timedelta(minutes=5)
    dec_time = df.index + step

    m15 = resample(df, "15min")
    box = m15_sideway_boxes(m15, cfg)
    pos = np.searchsorted(m15.index.values, dec_time.values, side="right") - 1
    bpos = pos - 2  # baseShift = 2
    valid = bpos >= 0
    bp = np.clip(bpos, 0, None)
    box_ok = valid & box["ok"].to_numpy()[bp]
    box_hi = box["hi"].to_numpy()[bp]
    box_lo = box["lo"].to_numpy()[bp]

    h1 = resample(df, "1h")
    a1 = atr(h1, 14).to_numpy()
    hp = np.searchsorted(h1.index.values, dec_time.values, side="right") - 1
    now = np.where(hp - 1 >= 0, a1[np.clip(hp - 1, 0, None)], np.nan)
    ref = np.where(hp - 24 >= 0, a1[np.clip(hp - 24, 0, None)], np.nan)
    # EA: if(atr_avg <= 0) return true;  -> NaN (no history) also passes
    vol_ok = ~((ref > 0) & (now > ref * cfg.vol_breaker))

    hhmm = (dec_time.hour * 100 + dec_time.minute).to_numpy()
    day = dec_time.values.astype("datetime64[D]").astype(np.int64)

    return Market(df, o, h, l, c, spread, times, dec_time, e, a, avg_body, sig, sw_hi, sw_lo,
                  last_sig, box_ok, box_hi, box_lo, vol_ok, hhmm, day)


# ----------------------------------------------------------------------------------------
# Signal helpers (direct ports)
# ----------------------------------------------------------------------------------------
def session_at(hhmm: int, cfg: EAConfig) -> int:
    for i, (s, e) in enumerate(cfg.sessions, start=1):
        if s <= hhmm < e:
            return i
    return 0


def session_time_left_pct(sess: int, hhmm: int, cfg: EAConfig) -> float:
    s, e = cfg.sessions[sess - 1]
    sm, em = (s // 100) * 60 + s % 100, (e // 100) * 60 + e % 100
    tot = em - sm
    if tot <= 0:
        return 0.0
    now = (hhmm // 100) * 60 + hhmm % 100
    return min(max(em - now, 0), tot) / tot * 100.0


def clamp_sl_dist(dist: float, atr1: float, cfg: EAConfig) -> float:
    """SSS_ClampSLDist(). NB: below the min it falls back to ATR*mult, which can still be < min."""
    if dist < cfg.p(cfg.sl_min_pts):
        dist = cfg.p(cfg.sl_min_pts) if cfg.sl_min_enforced else atr1 * cfg.sl_atr_fallback_mult
    if dist > cfg.p(cfg.sl_max_pts):
        dist = cfg.p(cfg.sl_max_pts)
    return dist


def pullback_scan(m: Market, k: int, t: int, max_wait: int):
    """Inner loop of SSS_CheckPullbackEntry / CheckSSSRejectRebound for breakout bar k,
    evaluated when bar t has just closed. Returns (dir, pullback_extreme, pullback_bar) or None."""
    d = int(m.sig[k])
    swL, swH = m.sw_lo[k], m.sw_hi[k]
    pb, ext = -1, 0.0
    for j in range(k + 1, t + 1):
        if j - k > max_wait:
            return None
        e = m.ema[j]
        if d == 1:
            if m.l[j] < swL:
                return None
            if pb < 0:
                if m.l[j] <= e:
                    pb, ext = j, m.l[j]
                continue
            ext = min(ext, m.l[j])
            if m.c[j] > e and m.c[j] > m.o[j]:
                return (1, ext, pb) if j == t else None
        else:
            if m.h[j] > swH:
                return None
            if pb < 0:
                if m.h[j] >= e:
                    pb, ext = j, m.h[j]
                continue
            ext = max(ext, m.h[j])
            if m.c[j] < e and m.c[j] < m.o[j]:
                return (-1, ext, pb) if j == t else None
    return None


def is_spike(m: Market, t: int, cfg: EAConfig) -> bool:
    b = abs(m.c[t] - m.o[t])
    ab, a = m.avg_body[t], m.atr[t]
    if not ab or ab <= 0 or np.isnan(ab):
        return False
    return (b / ab > cfg.spike_body_ratio) and (a > 0 and b / a > 2.0)


def entry_sl(m: Market, d: int, ext: float, t: int, cfg: EAConfig):
    """SL for a pullback/rebound entry at open[t+1]. Returns (entry, sl, raw_dist, clamped_dist)."""
    bid = m.o[t + 1]
    ask = bid + m.spread[t + 1]
    buf = cfg.p(cfg.sl_fixed_buffer_pts)
    if d == 1:
        raw = abs(ask - (ext - buf))
        dist = clamp_sl_dist(raw, m.atr[t], cfg)
        return ask, ask - dist, raw, dist
    raw = abs(bid - (ext + buf))
    dist = clamp_sl_dist(raw, m.atr[t], cfg)
    return bid, bid + dist, raw, dist


# ----------------------------------------------------------------------------------------
# Trade management simulation (ManageAutoSL + Double-Reject + News protection)
# ----------------------------------------------------------------------------------------
def simulate_set(m: Market, t: int, d: int, entry: float, sl: float, cfg: EAConfig, send_tp2: bool = True):
    """Simulate the TP1 (+TP2) order-set opened at open[t+1]. Bar-level, conservative:
    when SL and TP are both inside one bar, SL is assumed first; SL modifications computed
    from a bar's extreme take effect from the next bar."""
    n = len(m.c)
    e0 = t + 1
    R = abs(entry - sl)
    tp1 = entry + d * R * cfg.rr1
    tp2 = entry + d * R * cfg.rr2
    legs = {"tp1": dict(open=True, sl=sl, tp=tp1, exit=None, reason=None, bar=None, at_open=False)}
    if send_tp2 and cfg.rr2 > 0:
        legs["tp2"] = dict(open=True, sl=sl, tp=tp2, exit=None, reason=None, bar=None, at_open=False,
                           base_sl=None, done=False)
    band = cfg.p(cfg.dr_band_pts)
    tp1_dist = abs(tp1 - entry)
    lock_level = entry + d * tp1_dist * cfg.dr_lock_pct / 100.0
    dr_level, dr_touch, dr_count = 0.0, False, 0
    # EA seeds drPeak with Bid (BUY) / Ask (SELL) when it first sees the order
    dr_peak = entry - m.spread[e0] if d == 1 else entry
    news_h, news_m = cfg.news_hhmm // 100, cfg.news_hhmm % 100
    news_done_day = None

    def close_leg(leg, price, j, reason, at_open):
        leg.update(open=False, exit=price, reason=reason, bar=j, at_open=at_open)

    last = min(n - 1, e0 + cfg.max_hold_bars)
    for j in range(e0, last + 1):
        spr = m.spread[j]
        bid_o = m.o[j]
        px_o = bid_o if d == 1 else bid_o + spr   # closing price at open (Bid for buy, Ask for sell)

        # --- Double-Reject Close: evaluated on the first tick of bar j using bar j-1 ---
        if cfg.dr_enabled and any(L["open"] for L in legs.values()):
            hi1, lo1 = m.h[j - 1], m.l[j - 1]
            closed_dr = False
            if d == 1:
                if dr_level <= 0:
                    dr_peak = max(dr_peak, hi1)
                    if dr_peak >= lock_level:
                        dr_level = dr_peak
                elif hi1 >= dr_level + band:
                    dr_level, dr_touch, dr_peak = 0.0, False, hi1
                elif lo1 <= entry - band:
                    if not dr_touch:
                        dr_count += 1
                        dr_touch = True
                        closed_dr = dr_count >= cfg.dr_close_count
                else:
                    dr_touch = False
            else:
                if dr_level <= 0:
                    dr_peak = min(dr_peak, lo1)
                    if dr_peak <= lock_level:
                        dr_level = dr_peak
                elif lo1 <= dr_level - band:
                    dr_level, dr_touch, dr_peak = 0.0, False, lo1
                elif hi1 >= entry + band:
                    if not dr_touch:
                        dr_count += 1
                        dr_touch = True
                        closed_dr = dr_count >= cfg.dr_close_count
                else:
                    dr_touch = False
            if closed_dr:
                for L in legs.values():
                    if L["open"]:
                        close_leg(L, px_o, j, "double_reject", True)
                break

        # --- News protection: first tick inside [news-5m, news) once per day ---
        if cfg.news_enabled:
            bt = pd.Timestamp(m.times[j])
            nt = bt.normalize() + pd.Timedelta(hours=news_h, minutes=news_m)
            if nt - pd.Timedelta(minutes=cfg.news_mins_before) <= bt < nt and news_done_day != bt.normalize():
                news_done_day = bt.normalize()
                for L in legs.values():
                    if not L["open"]:
                        continue
                    dsl, dtp = abs(entry - L["sl"]), abs(L["tp"] - entry)
                    if dsl <= 0 or dtp <= 0:
                        continue
                    to_sl = (entry - px_o) if d == 1 else (px_o - entry)
                    to_tp = -to_sl
                    if to_sl > dsl * cfg.news_close_band_pct / 100:
                        continue
                    if to_tp <= dtp * cfg.news_close_band_pct / 100:
                        close_leg(L, px_o, j, "news_close", True)
                    else:
                        lk = entry + d * dtp * cfg.news_lock_pct / 100
                        if (d == 1 and lk > L["sl"]) or (d == -1 and lk < L["sl"]):
                            L["sl"] = lk

        # --- SL / TP hits inside bar j (SL first = conservative) ---
        lo_c = m.l[j] if d == 1 else m.l[j] + spr   # worst/best closing price in bar
        hi_c = m.h[j] if d == 1 else m.h[j] + spr
        for L in legs.values():
            if not L["open"]:
                continue
            if d == 1:
                if lo_c <= L["sl"]:
                    close_leg(L, min(L["sl"], px_o), j, "sl", px_o <= L["sl"])
                elif hi_c >= L["tp"]:
                    close_leg(L, max(L["tp"], px_o), j, "tp", px_o >= L["tp"])
            else:
                if hi_c >= L["sl"]:
                    close_leg(L, max(L["sl"], px_o), j, "sl", px_o >= L["sl"])
                elif lo_c <= L["tp"]:
                    close_leg(L, min(L["tp"], px_o), j, "tp", px_o <= L["tp"])
        if not any(L["open"] for L in legs.values()):
            break

        # --- Auto SL (lock profit) using this bar's best price, effective next bar ---
        if cfg.autosl_enabled:
            best = hi_c if d == 1 else lo_c
            L1 = legs["tp1"]
            if L1["open"]:
                tpd = abs(L1["tp"] - entry)
                prog = (best - entry) * d
                lp = 0.0
                if prog >= tpd * cfg.autosl_trig2 / 100:
                    lp = cfg.autosl_lock2
                elif prog >= tpd * cfg.autosl_trig1 / 100:
                    lp = cfg.autosl_lock1
                if lp > 0:
                    new = entry + d * tpd * lp / 100
                    if (new - L1["sl"]) * d > 0:
                        L1["sl"] = new
                        L2 = legs.get("tp2")
                        if L2 and L2["open"] and (new - L2["sl"]) * d > 0:
                            L2["sl"] = new
            L2 = legs.get("tp2")
            if L2 and L2["open"] and not L1["open"] and not L2["done"]:
                if L2["base_sl"] is None:
                    L2["base_sl"] = L2["sl"]
                full = (L2["tp"] - L2["base_sl"]) * d
                prog = (best - L2["base_sl"]) * d
                if full > 0:
                    lp, tier2 = 0.0, False
                    if prog >= full * cfg.tp2_trig2 / 100:
                        lp, tier2 = cfg.tp2_lock2, True
                    elif prog >= full * cfg.tp2_trig1 / 100:
                        lp = cfg.tp2_lock1
                    if lp > 0:
                        new = L2["base_sl"] + d * full * lp / 100
                        if (new - L2["sl"]) * d > 0:
                            L2["sl"] = new
                            if tier2:
                                L2["done"] = True

    for L in legs.values():
        if L["open"]:  # timeout
            j = last
            px = m.c[j] if d == 1 else m.c[j] + m.spread[j]
            L.update(open=False, exit=px, reason="timeout", bar=j, at_open=False)

    out = {"R_price": R, "tp1_price": tp1, "tp2_price": tp2 if "tp2" in legs else np.nan}
    r_set, exit_key = 0.0, 0
    for name, L in legs.items():
        r = (L["exit"] - entry) * d / R if R > 0 else 0.0
        out[f"{name}_r"] = r
        out[f"{name}_reason"] = L["reason"]
        out[f"{name}_bars"] = L["bar"] - e0
        r_set += 0.5 * r  # EA splits the base lot ~50/50 between TP1 and TP2
        exit_key = max(exit_key, L["bar"] * 2 + (0 if L["at_open"] else 1))
    if "tp2" not in legs:
        out.update(tp2_r=np.nan, tp2_reason=None, tp2_bars=np.nan)
    out["set_r"] = r_set
    out["exit_key"] = exit_key
    return out


# ----------------------------------------------------------------------------------------
# 1) Raw candidate scan: every bar where SSS_CheckPullbackEntry() would return a signal
# ----------------------------------------------------------------------------------------
def scan_pullback_candidates(m: Market, cfg: EAConfig):
    n = len(m.c)
    max_lb = cfg.pullback_max_wait + cfg.swing_bars + 5
    used_k = -1
    rows = []
    for t in range(40, n - 1):
        k = m.last_sig[t - 2]
        if k < 0 or k < t - max_lb + 1 or k <= used_k:
            continue
        res = pullback_scan(m, k, t, cfg.pullback_max_wait)
        if res is None:
            continue
        used_k = k
        d, ext, pb = res
        rows.append(dict(t=t, k=k, dir=d, pb_extreme=ext, pb_bar=pb))
    return rows


# ----------------------------------------------------------------------------------------
# 2) EA-faithful pass: session / cooldown / spike / S-R block / Reject+Rebound / guards
# ----------------------------------------------------------------------------------------
class SRBlock:
    """SSS_IsBlockedBySRZone() with its persisted state (side effects included, as in the EA)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.type, self.level, self.height = 0, 0.0, 0.0

    def check(self, m: Market, t: int, sig: int, price: float) -> bool:
        cfg = self.cfg
        if sig == self.type and self.level > 0:
            expired = cfg.block_expire_pct > 0 and self.height > 0 and \
                abs(price - self.level) > self.height * cfg.block_expire_pct / 100
            if expired:
                self.reset()
            elif sig == 1 and price >= self.level:
                return True
            elif sig == -1 and price <= self.level:
                return True
        if not m.box_ok[t]:
            return False
        hi, lo = m.box_hi[t], m.box_lo[t]
        zone = (hi - lo) * cfg.box_zone_pct / 100
        if sig == 1 and price >= hi - zone:
            self.type, self.level, self.height = 1, hi - zone, hi - lo
            return True
        if sig == -1 and price <= lo + zone:
            self.type, self.level, self.height = -1, lo + zone, hi - lo
            return True
        return False


def live_signal(m: Market, t: int) -> int:
    return int(m.sig[t])


def run_ea(m: Market, cfg: EAConfig, candidates):
    """Walk forward bar by bar like OnTick() on each new bar. Returns (trades, status_by_t)."""
    n = len(m.c)
    cand_by_t = {r["t"]: r for r in candidates}
    status = {}
    trades = []
    sr = SRBlock(cfg)
    sess_done = {}
    cur_day = None
    last_block_t = None
    last_rr_k = -1
    max_lb_rr = cfg.rebound_max_wait + cfg.swing_bars + 5

    def open_orders(dbar, sess=None):
        cnt = 0
        for tr in trades:
            if tr["exit_key"] > 2 * dbar and (sess is None or tr["session"] == sess):
                cnt += 2 if tr["tp2_sent"] else 1
        return cnt

    def closed_sets_today(dbar, sess, day):
        return sum(1 for tr in trades if tr["session"] == sess and tr["day"] == day and tr["exit_key"] <= 2 * dbar)

    def place(t, d, entry, sl, raw, dist, sess, kind, k, ext, pb, send_tp2):
        sim = simulate_set(m, t, d, entry, sl, cfg, send_tp2)
        tr = dict(t=t, k=k, dir=d, entry=entry, sl=sl, sl_raw_dist=raw, sl_dist=dist, session=sess,
                  day=int(m.day[t]), kind=kind, pb_extreme=ext, pb_bar=pb, tp2_sent=send_tp2,
                  time=m.dec_time[t])
        tr.update(sim)
        trades.append(tr)
        return tr

    for t in range(40, n - 1):
        dbar = t + 1
        day = int(m.day[t])
        if day != cur_day:
            cur_day = day
            sess_done = {}
            sr.reset()
        if not m.vol_ok[t]:
            if t in cand_by_t:
                status[t] = "vol_guard"
            continue
        sess = session_at(int(m.hhmm[t]), cfg)
        main_blocked_sr = False
        if sess > 0:
            has_open = open_orders(dbar, sess) > 0
            cnt = closed_sets_today(dbar, sess, day)
            traded = sess_done.get(sess, False) or has_open or cnt > 0
            can_re = (traded and not has_open and cfg.max_sets_per_session > 1 and 0 < cnt < cfg.max_sets_per_session
                      and session_time_left_pct(sess, int(m.hhmm[t]), cfg) > cfg.reentry_min_time_left_pct)
            session_traded = (traded and not can_re) or has_open
            if session_traded:
                if t in cand_by_t:
                    status[t] = "session_done"
            else:
                in_cd = last_block_t is not None and (t - last_block_t) < cfg.block_cooldown_bars
                if in_cd:
                    if t in cand_by_t:
                        status[t] = "cooldown"
                elif t in cand_by_t:
                    c = cand_by_t[t]
                    d = c["dir"]
                    if is_spike(m, t, cfg):
                        status[t] = "blocked_spike"
                    elif cfg.rebound_enabled and sr.check(m, t, d, m.c[t]):
                        status[t] = "blocked_sr"
                        last_block_t = t
                        main_blocked_sr = True
                    else:
                        entry, sl, raw, dist = entry_sl(m, d, c["pb_extreme"], t, cfg)
                        send_tp2 = cfg.rr2 > 0 and open_orders(dbar) + 2 <= cfg.max_orders
                        place(t, d, entry, sl, raw, dist, sess, "pullback", c["k"], c["pb_extreme"], c["pb_bar"], send_tp2)
                        sess_done[sess] = True
                        status[t] = "traded"
        elif t in cand_by_t:
            status[t] = "out_of_session"

        # --- Reject + Rebound (runs every bar while in a session, NOT gated by session_traded) ---
        if cfg.rebound_enabled and not cfg.rebound_trades_disabled and sess > 0 and open_orders(dbar) < cfg.max_orders:
            for ss in range(3, max_lb_rr + 1):
                k = t - ss + 1
                if k < 0 or m.sig[k] == 0:
                    continue
                d = int(m.sig[k])
                if not sr.check(m, t, d, m.c[k]):
                    continue
                if k <= last_rr_k:
                    break
                res = pullback_scan(m, k, t, cfg.rebound_max_wait)
                if res is not None:
                    _, ext, pb = res
                    if live_signal(m, t) == -d:
                        break
                    if m.box_ok[t] and ((d == 1 and m.c[t] <= m.box_lo[t]) or (d == -1 and m.c[t] >= m.box_hi[t])):
                        break
                    rr_sess = session_at(int(pd.Timestamp(m.times[k]).hour * 100 + pd.Timestamp(m.times[k]).minute), cfg)
                    last_rr_k = k
                    if rr_sess > 0:
                        entry, sl, raw, dist = entry_sl(m, d, ext, t, cfg)
                        tr = place(t, d, entry, sl, raw, dist, rr_sess, "reject_rebound", k, ext, pb, cfg.rr2 > 0)
                        tr["same_bar_as_sr_block"] = main_blocked_sr
                        sess_done[rr_sess] = True
                        if t in cand_by_t:
                            status[t] = "traded_rr"
                break
    return trades, status
