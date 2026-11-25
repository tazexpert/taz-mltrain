import os

# Limit OpenBLAS and OpenMP threads BEFORE importing numpy, pandas, or lightgbm
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

# ==========================================
# Global Simulation Parameters (TAZ v10.13 - No DIV)
# ==========================================
M1_PATH = "XAUUSD_M1.csv"
M5_PATH = "XAUUSD_M5.csv"

SPREAD = 0.25
BE_BUFFER = 0.5          # MPS_BEBufferPts / HTF_SLBufferPts
FRI_NO_ENTRY_MIN = 1200  # 20:00 Friday
FRI_CLOSE_MIN = 1305     # 21:45 Friday
N_FOLDS = 5              # 5-Fold Expanding Walk-Forward Validation


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Wilder's smoothed ATR matching MT4 iATR()."""
    prev_c = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs())
    )
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's smoothed RSI matching MT4 iRSI()."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Wilder's ADX(14) matching MT4 iADX()."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = wilder_atr(df, period)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def build_zigzag_pivots(df: pd.DataFrame, atr_col: str, mult: float = 1.5):
    """Non-repainting ZigZag matching HTF_ProcessBar."""
    highs = df["high"].values
    lows = df["low"].values
    atrs = df[atr_col].values

    pivots = []
    zz_up = True
    ext_price = highs[0]
    ext_idx = 0

    for i in range(1, len(df)):
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue
        if zz_up:
            if highs[i] > ext_price:
                ext_price = highs[i]
                ext_idx = i
            if ext_price - lows[i] >= mult * atr:
                pivots.append({"is_high": True, "price": ext_price, "pivot_idx": ext_idx, "confirm_idx": i})
                zz_up = False
                ext_price = lows[i]
                ext_idx = i
        else:
            if lows[i] < ext_price:
                ext_price = lows[i]
                ext_idx = i
            if highs[i] - ext_price >= mult * atr:
                pivots.append({"is_high": False, "price": ext_price, "pivot_idx": ext_idx, "confirm_idx": i})
                zz_up = True
                ext_price = highs[i]
                ext_idx = i
    return pivots


def load_data():
    """Load M1 and M5 data and compute indicators for MPS, AIP, and HTF_ASIA."""
    print("Loading M1 and M5 CSV files...")
    m1 = pd.read_csv(
        M1_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m5 = pd.read_csv(
        M5_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    # 1. H1 Resampling & Indicators
    h1 = m5.set_index("time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    h1["h1_ema50"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_ema200"] = h1["close"].ewm(span=200, adjust=False).mean()
    h1["h1_rsi14"] = wilder_rsi(h1["close"], 14)
    h1["h1_atr14"] = wilder_atr(h1, 14)
    h1["h1_ready_time"] = h1["time"] + pd.Timedelta(hours=1)
    h1_r = h1[["h1_ready_time", "close", "h1_ema50", "h1_ema200", "h1_rsi14"]].rename(
        columns={"close": "h1_close"}
    )

    # 2. M5 Indicators
    m5["ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
    m5["ema50"] = m5["close"].ewm(span=50, adjust=False).mean()
    m5["ema200"] = m5["close"].ewm(span=200, adjust=False).mean()

    m5["adx14"] = compute_adx(m5, 14)
    m5["w_atr14"] = wilder_atr(m5, 14)
    m5["w_rsi14"] = wilder_rsi(m5["close"], 14)

    # MT4 iMACD (EMA12 - EMA26, Signal = SMA9)
    ema12 = m5["close"].ewm(span=12, adjust=False).mean()
    ema26 = m5["close"].ewm(span=26, adjust=False).mean()
    m5["macd"] = ema12 - ema26
    m5["macd_sig"] = m5["macd"].rolling(9).mean()
    m5["macd_hist"] = m5["macd"] - m5["macd_sig"]

    # AIP_SimpleATR & AIP_SimpleRSI
    prev_c = m5["close"].shift(1)
    tr = np.maximum(m5["high"] - m5["low"], np.maximum((m5["high"] - prev_c).abs(), (m5["low"] - prev_c).abs()))
    m5["s_atr14"] = tr.rolling(14).mean()

    delta = m5["close"].diff()
    s_gain = delta.clip(lower=0).rolling(14).mean()
    s_loss = (-delta.clip(upper=0)).rolling(14).mean()
    m5["s_rsi14"] = 100.0 - (100.0 / (1.0 + s_gain / s_loss.replace(0, np.nan)))

    rng = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body_ratio"] = (m5["close"] - m5["open"]) / rng
    m5["hist_norm"] = m5["macd_hist"] / m5["s_atr14"]

    m5["entry_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["hour"] = m5["entry_time"].dt.hour
    m5["minute"] = m5["entry_time"].dt.minute
    m5["hhmm"] = m5["hour"] * 100 + m5["minute"]
    m5["dayofweek"] = m5["entry_time"].dt.dayofweek
    m5["min_of_day"] = m5["hour"] * 60 + m5["minute"]

    m5 = pd.merge_asof(m5, h1_r, left_on="entry_time", right_on="h1_ready_time", direction="backward")
    m5 = m5[(m5["entry_time"] >= m1["time"].min()) & (m5["entry_time"] <= m1["time"].max())].dropna().reset_index(drop=True)

    return m1, m5, h1


def generate_strategy_signals(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Generate unified signal table across MPS, AIP (Multi-Setup), and HTF_ASIA."""
    signals = []
    fri_no_entry = (m5["dayofweek"] == 4) & (m5["min_of_day"] >= FRI_NO_ENTRY_MIN)

    # ---------------------------------------------------------
    # 1. MPS (Momentum Pullback Scalper)
    # ---------------------------------------------------------
    ema_sep = (m5["ema21"] - m5["ema50"]).abs() / m5["w_atr14"]
    dist200_buy = (m5["close"] - m5["ema200"]) / m5["w_atr14"]
    dist200_sell = (m5["ema200"] - m5["close"]) / m5["w_atr14"]

    win_a = (m5["hhmm"] >= 230) & (m5["hhmm"] <= 729)
    win_b = (m5["hhmm"] >= 730) & (m5["hhmm"] <= 959)
    win_c = (m5["hhmm"] >= 1230) & (m5["hhmm"] <= 1259)
    win_d = (m5["min_of_day"] >= 1320) | (m5["min_of_day"] < 180)

    mps_buy_raw = (m5["h1_rsi14"] > 50.0) & (m5["ema21"] < m5["ema50"])
    mps_sell_raw = (m5["h1_rsi14"] < 45.0) & (m5["w_rsi14"] > 50.0)
    mps_ambig = mps_buy_raw & mps_sell_raw

    mps_buy = mps_buy_raw & (~mps_ambig) & (ema_sep >= 0.10) & (dist200_buy <= 5.0) & (~win_a) & (~win_c) & (~win_d) & (~fri_no_entry)
    mps_sell = mps_sell_raw & (~mps_ambig) & (ema_sep >= 0.10) & (dist200_sell <= 5.0) & (~win_b) & (~win_c) & (~win_d) & (~fri_no_entry)

    for idx in np.where(mps_buy | mps_sell)[0]:
        direction = 1 if mps_buy.iloc[idx] else -1
        sl_dist = float(np.clip(m5["w_atr14"].iloc[idx] * 1.0, 6.5, 9.0))
        signals.append({
            "entry_time": m5["entry_time"].iloc[idx],
            "strategy": "MPS", "slot": "MPS",
            "dir": direction, "sl_dist": sl_dist, "rr1": 1.0, "rr2": 2.5,
            "max_hold_min": 288 * 5, "daily_close_min": -1
        })

    # ---------------------------------------------------------
    # 2. AIP (Multi-Setup: InsideBar + MACD + EMA9/20)
    # ---------------------------------------------------------
    adx_ok = m5["adx14"] >= 18.0
    h1_up = (m5["h1_close"] > m5["h1_ema50"]) & (m5["h1_ema50"] > m5["h1_ema200"])
    h1_dn = (m5["h1_close"] < m5["h1_ema50"]) & (m5["h1_ema50"] < m5["h1_ema200"])

    ib = (m5["high"].shift(1) < m5["high"].shift(2)) & (m5["low"].shift(1) > m5["low"].shift(2))
    ib_b = (~fri_no_entry) & adx_ok & h1_up & ib & (m5["close"] > m5["high"].shift(2))
    ib_s = (~fri_no_entry) & adx_ok & h1_dn & ib & (m5["close"] < m5["low"].shift(2))

    mc_b = (~fri_no_entry) & adx_ok & h1_up & (m5["macd"] > m5["macd_sig"]) & (m5["macd"].shift(1) <= m5["macd_sig"].shift(1)) & (m5["macd"] < 0)
    mc_s = (~fri_no_entry) & adx_ok & h1_dn & (m5["macd"] < m5["macd_sig"]) & (m5["macd"].shift(1) >= m5["macd_sig"].shift(1)) & (m5["macd"] > 0)

    em_b = (~fri_no_entry) & adx_ok & h1_up & (m5["ema9"] > m5["ema20"]) & (m5["ema9"].shift(1) <= m5["ema20"].shift(1))
    em_s = (~fri_no_entry) & adx_ok & h1_dn & (m5["ema9"] < m5["ema20"]) & (m5["ema9"].shift(1) >= m5["ema20"].shift(1))

    aip_any = ib_b | ib_s | mc_b | mc_s | em_b | em_s
    for idx in np.where(aip_any)[0]:
        if ib_b.iloc[idx]: d = 1
        elif ib_s.iloc[idx]: d = -1
        elif mc_b.iloc[idx]: d = 1
        elif mc_s.iloc[idx]: d = -1
        elif em_b.iloc[idx]: d = 1
        else: d = -1
        sl_dist = float(m5["s_atr14"].iloc[idx] * 1.5)
        if sl_dist <= 0:
            continue
        signals.append({
            "entry_time": m5["entry_time"].iloc[idx],
            "strategy": "AIP", "slot": "AIP",
            "dir": d, "sl_dist": sl_dist, "rr1": 1.0, "rr2": 2.5,
            "max_hold_min": 240, "daily_close_min": -1
        })

    # ---------------------------------------------------------
    # 3. HTF_ASIA (Asia Range Breakout on H1)
    # ---------------------------------------------------------
    h1_sub = h1[(h1["h1_ready_time"] >= m5["entry_time"].min()) & (h1["h1_ready_time"] <= m5["entry_time"].max())].dropna().reset_index(drop=True)
    h1_pivots = build_zigzag_pivots(h1_sub, "h1_atr14", mult=1.5)
    h1_sub["date"] = h1_sub["time"].dt.date
    h1_sub["hour"] = h1_sub["time"].dt.hour

    asia_ranges = h1_sub[(h1_sub["hour"] >= 0) & (h1_sub["hour"] <= 6)].groupby("date").agg(
        asia_hi=("high", "max"), asia_lo=("low", "min"), bars=("high", "count")
    )
    used_day_side = set()

    for i in range(len(h1_sub)):
        hr = h1_sub["hour"].iloc[i]
        if hr < 7 or hr > 15:
            continue
        dt = h1_sub["date"].iloc[i]
        if dt not in asia_ranges.index or asia_ranges.loc[dt, "bars"] < 3:
            continue
        a_hi = asia_ranges.loc[dt, "asia_hi"]
        a_lo = asia_ranges.loc[dt, "asia_lo"]
        c1 = h1_sub["close"].iloc[i]
        if c1 > a_hi:
            d = 1
        elif c1 < a_lo:
            d = -1
        else:
            continue

        if (dt, d) in used_day_side:
            continue
        used_day_side.add((dt, d))

        valid_pivs = [p for p in h1_pivots if p["confirm_idx"] <= i]
        if not valid_pivs:
            continue
        last_c = valid_pivs[-1]["confirm_idx"]
        entry_px = c1 + (SPREAD if d == 1 else 0.0)
        leg_ext = h1_sub["low"].iloc[last_c:i + 1].min() if d == 1 else h1_sub["high"].iloc[last_c:i + 1].max()
        ref = None
        if (d == 1 and leg_ext < entry_px - 0.50) or (d == -1 and leg_ext > entry_px + 0.50):
            ref = leg_ext
        else:
            for p in reversed(valid_pivs[-24:]):
                if (d == 1 and (not p["is_high"]) and p["price"] < entry_px - 0.50) or (d == -1 and p["is_high"] and p["price"] > entry_px + 0.50):
                    ref = p["price"]
                    break
        if ref is None:
            continue

        sl_dist = float(np.clip(abs(entry_px - (ref - 0.50 if d == 1 else ref + 0.50)), 15.0, 45.0))
        signals.append({
            "entry_time": h1_sub["h1_ready_time"].iloc[i],
            "strategy": "HTF_ASIA", "slot": "HTF",
            "dir": d, "sl_dist": sl_dist, "rr1": 1.0, "rr2": 3.0,
            "max_hold_min": 24 * 60, "daily_close_min": 1315
        })

    return pd.DataFrame(signals).sort_values("entry_time").reset_index(drop=True)


def extract_all_snapshots(m1_arrays, m5_lookup, sig_df):
    """Simulate every signal on M1 and record strategy-agnostic M5 in-trade snapshots."""
    m1_times, m1_opens, m1_highs, m1_lows, m1_closes, m1_dow, m1_mod = m1_arrays
    m5_times, m5_adx, m5_rsi, m5_hist, m5_body = m5_lookup
    n_m1 = len(m1_times)

    snapshots = []
    trade_meta = []

    for tid, r in sig_df.iterrows():
        entry_time = r["entry_time"].to_datetime64()
        direction = int(r["dir"])
        sl_dist = float(r["sl_dist"])
        rr1, rr2 = float(r["rr1"]), float(r["rr2"])
        max_hold_min = int(r["max_hold_min"])
        daily_close_min = int(r["daily_close_min"])

        s_idx = np.searchsorted(m1_times, entry_time, side="left")
        if s_idx >= n_m1 - 1:
            continue

        max_exit_time = entry_time + np.timedelta64(max_hold_min, "m")
        entry = (m1_opens[s_idx] + SPREAD) if direction == 1 else m1_opens[s_idx]
        sl_p = entry - direction * sl_dist
        tp1_p = entry + direction * sl_dist * rr1
        tp2_p = entry + direction * sl_dist * rr2
        o2_sl = sl_p

        hit_tp1 = False
        mfe_r, mae_r = 0.0, 0.0
        final_r, exit_time = np.nan, pd.NaT
        trade_snaps = []

        for j in range(s_idx, n_m1):
            t_now = m1_times[j]
            if direction == 1:
                fav_r = (m1_highs[j] - entry) / sl_dist
                adv_r = (m1_lows[j] - entry) / sl_dist
                cur_px = m1_closes[j]
                if not hit_tp1:
                    if m1_lows[j] <= sl_p:
                        final_r, exit_time = -2.0, t_now
                        break
                    if m1_highs[j] >= tp1_p:
                        hit_tp1 = True
                        o2_sl = entry + BE_BUFFER
                        if m1_highs[j] >= tp2_p:
                            final_r, exit_time = rr1 + rr2, t_now
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        final_r, exit_time = rr1 + (o2_sl - entry) / sl_dist, t_now
                        break
                    if m1_highs[j] >= tp2_p:
                        final_r, exit_time = rr1 + rr2, t_now
                        break
            else:
                ah, al = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                fav_r = (entry - al) / sl_dist
                adv_r = (entry - ah) / sl_dist
                cur_px = m1_closes[j] + SPREAD
                if not hit_tp1:
                    if ah >= sl_p:
                        final_r, exit_time = -2.0, t_now
                        break
                    if al <= tp1_p:
                        hit_tp1 = True
                        o2_sl = entry - BE_BUFFER
                        if al <= tp2_p:
                            final_r, exit_time = rr1 + rr2, t_now
                            break
                else:
                    if ah >= o2_sl:
                        final_r, exit_time = rr1 + (entry - o2_sl) / sl_dist, t_now
                        break
                    if al <= tp2_p:
                        final_r, exit_time = rr1 + rr2, t_now
                        break

            mfe_r = max(mfe_r, fav_r)
            mae_r = min(mae_r, adv_r)

            force_daily = (daily_close_min > 0) and (m1_mod[j] >= daily_close_min)
            force_fri = (m1_dow[j] == 4) and (m1_mod[j] >= FRI_CLOSE_MIN)
            if force_daily or force_fri or (t_now >= max_exit_time):
                leg_r = direction * (cur_px - entry) / sl_dist
                final_r = (rr1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                exit_time = t_now
                break

            # M5 closed bar snapshot
            if m1_mod[j] % 5 == 4:
                next_m5_t = t_now + np.timedelta64(1, "m")
                m5_idx = np.searchsorted(m5_times, next_m5_t, side="right") - 1
                if m5_idx >= 0:
                    bars_held = int((next_m5_t - entry_time) / np.timedelta64(5, "m"))
                    leg_r = direction * (cur_px - entry) / sl_dist
                    curr_total_r = (rr1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                    tp_rem_r = (rr2 - leg_r) if hit_tp1 else (rr1 - leg_r)
                    drawback_r = mfe_r - leg_r

                    rsi_dir = m5_rsi[m5_idx] if direction == 1 else (100.0 - m5_rsi[m5_idx])
                    hist_dir = direction * m5_hist[m5_idx]
                    body_dir = direction * m5_body[m5_idx]

                    trade_snaps.append({
                        "snap_time": pd.Timestamp(next_m5_t),
                        "bars_held": bars_held,
                        "hit_tp1": int(hit_tp1),
                        "curr_r": leg_r,
                        "tp_rem_r": tp_rem_r,
                        "mfe_r": mfe_r,
                        "mae_r": mae_r,
                        "drawback_r": drawback_r,
                        "adx14": m5_adx[m5_idx],
                        "rsi_dir": rsi_dir,
                        "hist_dir": hist_dir,
                        "body_dir": body_dir,
                        "curr_total_r": curr_total_r
                    })

        if not np.isnan(final_r):
            trade_meta.append({
                "trade_id": tid, "entry_time": r["entry_time"],
                "strategy": r["strategy"], "slot": r["slot"],
                "dir": direction, "sl_dist": sl_dist, "rr1": rr1, "rr2": rr2,
                "max_hold_min": max_hold_min, "daily_close_min": daily_close_min,
                "base_r": final_r, "base_exit": pd.Timestamp(exit_time)
            })
            for s in trade_snaps:
                s["target_hold"] = 1 if final_r > s["curr_total_r"] else 0
                snapshots.append(s)

    return pd.DataFrame(trade_meta), pd.DataFrame(snapshots)


def simulate_portfolio(m1_arrays, m5_lookup, oos_trades, fold_models, mode="baseline", close_th=0.30, tighten_th=0.40):
    """Run realistic slot-isolated portfolio simulation across MPS, AIP, and HTF."""
    m1_times, m1_opens, m1_highs, m1_lows, m1_closes, m1_dow, m1_mod = m1_arrays
    m5_times, m5_adx, m5_rsi, m5_hist, m5_body = m5_lookup
    n_m1 = len(m1_times)

    slot_next_free = {"MPS": pd.Timestamp.min, "AIP": pd.Timestamp.min, "HTF": pd.Timestamp.min}
    mps_daily_sets = {}
    last_mps_sl_time = None
    records = []

    for _, r in oos_trades.sort_values("entry_time").iterrows():
        slot = r["slot"]
        strat = r["strategy"]
        et = r["entry_time"]

        # 1. Slot 1-set-at-a-time gate
        if et < slot_next_free[slot]:
            continue

        # 2. MPS daily cap (7 sets/day) & 60-min SL cooldown
        if slot == "MPS":
            day_key = et.date()
            if mps_daily_sets.get(day_key, 0) >= 7:
                continue
           
                # Change from: if (et - last_mps_sl_time) < pd.Timedelta(minutes=60):
        if last_mps_sl_time is not None and (et - last_mps_sl_time) < pd.Timedelta(minutes=60):
                continue

        if mode == "baseline":
            final_r = float(r["base_r"])
            exit_time = r["base_exit"]
        else:
            model = fold_models.get(int(r["fold"]), None)
            if model is None:
                final_r, exit_time = float(r["base_r"]), r["base_exit"]
            else:
                entry_t64 = et.to_datetime64()
                direction = int(r["dir"])
                sl_dist = float(r["sl_dist"])
                rr1, rr2 = float(r["rr1"]), float(r["rr2"])
                max_hold_min = int(r["max_hold_min"])
                daily_close_min = int(r["daily_close_min"])

                s_idx = np.searchsorted(m1_times, entry_t64, side="left")
                max_exit_time = entry_t64 + np.timedelta64(max_hold_min, "m")
                entry = (m1_opens[s_idx] + SPREAD) if direction == 1 else m1_opens[s_idx]
                sl_p = entry - direction * sl_dist
                tp1_p = entry + direction * sl_dist * rr1
                tp2_p = entry + direction * sl_dist * rr2
                o1_sl, o2_sl = sl_p, sl_p

                hit_tp1 = False
                mfe_r, mae_r = 0.0, 0.0
                final_r, exit_time = float(r["base_r"]), r["base_exit"]

                for j in range(s_idx, n_m1):
                    t_now = m1_times[j]
                    if direction == 1:
                        fav_r = (m1_highs[j] - entry) / sl_dist
                        adv_r = (m1_lows[j] - entry) / sl_dist
                        cur_px = m1_closes[j]
                        if not hit_tp1:
                            if m1_lows[j] <= o1_sl:
                                final_r, exit_time = 2.0 * (o1_sl - entry) / sl_dist, pd.Timestamp(t_now)
                                break
                            if m1_highs[j] >= tp1_p:
                                hit_tp1 = True
                                o2_sl = max(o2_sl, entry + BE_BUFFER)
                                if m1_highs[j] >= tp2_p:
                                    final_r, exit_time = rr1 + rr2, pd.Timestamp(t_now)
                                    break
                        else:
                            if m1_lows[j] <= o2_sl:
                                final_r, exit_time = rr1 + (o2_sl - entry) / sl_dist, pd.Timestamp(t_now)
                                break
                            if m1_highs[j] >= tp2_p:
                                final_r, exit_time = rr1 + rr2, pd.Timestamp(t_now)
                                break
                    else:
                        ah, al = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                        fav_r = (entry - al) / sl_dist
                        adv_r = (entry - ah) / sl_dist
                        cur_px = m1_closes[j] + SPREAD
                        if not hit_tp1:
                            if ah >= o1_sl:
                                final_r, exit_time = 2.0 * (entry - o1_sl) / sl_dist, pd.Timestamp(t_now)
                                break
                            if al <= tp1_p:
                                hit_tp1 = True
                                o2_sl = min(o2_sl, entry - BE_BUFFER)
                                if al <= tp2_p:
                                    final_r, exit_time = rr1 + rr2, pd.Timestamp(t_now)
                                    break
                        else:
                            if ah >= o2_sl:
                                final_r, exit_time = rr1 + (entry - o2_sl) / sl_dist, pd.Timestamp(t_now)
                                break
                            if al <= tp2_p:
                                final_r, exit_time = rr1 + rr2, pd.Timestamp(t_now)
                                break

                    mfe_r = max(mfe_r, fav_r)
                    mae_r = min(mae_r, adv_r)

                    force_daily = (daily_close_min > 0) and (m1_mod[j] >= daily_close_min)
                    force_fri = (m1_dow[j] == 4) and (m1_mod[j] >= FRI_CLOSE_MIN)
                    if force_daily or force_fri or (t_now >= max_exit_time):
                        leg_r = direction * (cur_px - entry) / sl_dist
                        final_r = (rr1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                        exit_time = pd.Timestamp(t_now)
                        break

                    # Universal Exit AI evaluation at closed M5 bar
                    if m1_mod[j] % 5 == 4:
                        next_m5_t = t_now + np.timedelta64(1, "m")
                        m5_idx = np.searchsorted(m5_times, next_m5_t, side="right") - 1
                        if m5_idx >= 0:
                            bars_held = int((next_m5_t - entry_t64) / np.timedelta64(5, "m"))
                            leg_r = direction * (cur_px - entry) / sl_dist
                            tp_rem_r = (rr2 - leg_r) if hit_tp1 else (rr1 - leg_r)
                            drawback_r = mfe_r - leg_r
                            rsi_dir = m5_rsi[m5_idx] if direction == 1 else (100.0 - m5_rsi[m5_idx])
                            hist_dir = direction * m5_hist[m5_idx]
                            body_dir = direction * m5_body[m5_idx]

                            X_snap = np.array([[
                                bars_held, int(hit_tp1), leg_r, tp_rem_r, mfe_r, mae_r, drawback_r,
                                m5_adx[m5_idx], rsi_dir, hist_dir, body_dir
                            ]], dtype=np.float32)
                            p_hold = model.predict_proba(X_snap)[0, 1]

                            if p_hold < close_th:
                                final_r = (rr1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                                exit_time = pd.Timestamp(t_now)
                                break
                            elif mode == "hybrid" and p_hold < tighten_th:
                                if direction == 1:
                                    target_sl = entry + BE_BUFFER if cur_px > entry + BE_BUFFER else (entry - 0.5 * sl_dist)
                                    if target_sl < cur_px:
                                        o1_sl = max(o1_sl, target_sl)
                                        o2_sl = max(o2_sl, target_sl)
                                else:
                                    target_sl = entry - BE_BUFFER if cur_px < entry - BE_BUFFER else (entry + 0.5 * sl_dist)
                                    if target_sl > cur_px:
                                        o1_sl = min(o1_sl, target_sl)
                                        o2_sl = min(o2_sl, target_sl)

        slot_next_free[slot] = exit_time
        if slot == "MPS":
            day_key = et.date()
            mps_daily_sets[day_key] = mps_daily_sets.get(day_key, 0) + 1
            if final_r <= -0.5:
                last_mps_sl_time = exit_time

        records.append({"strategy": strat, "r": final_r, "exit_time": exit_time})

    return pd.DataFrame(records)


def calc_stats(arr: np.ndarray):
    if len(arr) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    net_r = arr.sum()
    wr = (arr > 0).mean() * 100.0
    avg_r = arr.mean()
    gp, gl = arr[arr > 0].sum(), abs(arr[arr < 0].sum())
    pf = gp / gl if gl > 0 else 9.99
    eq = np.concatenate([[0.0], np.cumsum(arr)])
    mdd = (np.maximum.accumulate(eq) - eq).max()
    return len(arr), wr, avg_r, net_r, pf, mdd, (net_r / mdd if mdd > 0 else 9.99)


def main():
    m1, m5, h1 = load_data()
    m1_arrays = (
        m1["time"].values, m1["open"].values, m1["high"].values,
        m1["low"].values, m1["close"].values,
        m1["time"].dt.dayofweek.values, (m1["time"].dt.hour * 60 + m1["time"].dt.minute).values
    )
    m5_lookup = (
        m5["entry_time"].values, m5["adx14"].values, m5["s_rsi14"].values,
        m5["hist_norm"].values, m5["body_ratio"].values
    )

    print("Generating signals across MPS, AIP, and HTF_ASIA (DIV excluded)...")
    sig_df = generate_strategy_signals(m5, h1)
    print("Signal counts by strategy:", sig_df["strategy"].value_counts().to_dict())

    print("Extracting M5 in-trade snapshots...")
    trades_df, snaps_df = extract_all_snapshots(m1_arrays, m5_lookup, sig_df)
    print(f"Extracted {len(snaps_df):,} M5 snapshots from {len(trades_df):,} trades.")

    t_min, t_max = m5["entry_time"].min(), m5["entry_time"].max() + pd.Timedelta(minutes=1)
    time_edges = pd.date_range(start=t_min, end=t_max, periods=N_FOLDS + 2)

    feat_cols = [
        "bars_held", "hit_tp1", "curr_r", "tp_rem_r", "mfe_r", "mae_r", "drawback_r",
        "adx14", "rsi_dir", "hist_dir", "body_dir"
    ]

    print("Training 5-Fold Walk-Forward Universal Exit Manager AI...")
    fold_models = {}
    oos_list = []

    for fold in range(1, N_FOLDS + 1):
        tr_end, te_start, te_end = time_edges[fold], time_edges[fold], time_edges[fold + 1]
        tr_snaps = snaps_df[snaps_df["snap_time"] < tr_end]
        te_trades = trades_df[(trades_df["entry_time"] >= te_start) & (trades_df["entry_time"] < te_end)].copy()
        te_trades["fold"] = fold
        oos_list.append(te_trades)

        if len(tr_snaps) >= 300 and tr_snaps["target_hold"].nunique() == 2:
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=150, learning_rate=0.03,
                max_depth=3, num_leaves=7, min_child_samples=60,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, n_jobs=2, random_state=42, verbose=-1
            )
            model.fit(tr_snaps[feat_cols].values, tr_snaps["target_hold"].values)
            fold_models[fold] = model

    oos_trades = pd.concat(oos_list).sort_values("entry_time").reset_index(drop=True)

    print("\n" + "=" * 102)
    print("5-FOLD OOS: UNIVERSAL EXIT AI vs BASELINE (MPS + AIP + HTF_ASIA | NO DIV)")
    print("=" * 102)
    header = f"{'Exit Policy':34s} | {'Trades':>6s} | {'WinRate':>7s} | {'Avg R':>7s} | {'Net R':>8s} | {'PF':>5s} | {'MDD(R)':>7s} | {'Net/MDD':>7s}"
    print(header)
    print("-" * len(header))

    configs = [
        ("Baseline (Normal v10.13 Exit)", "baseline", 0.0, 0.0),
        ("AI Early Close (P_hold < 0.25)", "early_close", 0.25, 0.0),
        ("AI Early Close (P_hold < 0.30)", "early_close", 0.30, 0.0),
        ("AI Early Close (P_hold < 0.35)", "early_close", 0.35, 0.0),
        ("AI Hybrid (Tight<0.40, Close<0.25)", "hybrid", 0.25, 0.40),
    ]

    best_df = None
    base_df = None
    for label, mode, c_th, t_th in configs:
        res_df = simulate_portfolio(m1_arrays, m5_lookup, oos_trades, fold_models, mode, c_th, t_th)
        res_df = res_df.sort_values("exit_time")
        n, wr, ar, nr, pf, mdd, rm = calc_stats(res_df["r"].values)
        print(f"{label:34s} | {n:6d} | {wr:6.2f}% | {ar:+6.3f}R | {nr:+7.2f}R | {pf:5.2f} | {mdd:6.1f}R | {rm:+7.2f}")
        if mode == "baseline":
            base_df = res_df
        if label == "AI Hybrid (Tight<0.40, Close<0.25)":
            best_df = res_df

    print("\n" + "=" * 102)
    print("PER-STRATEGY BREAKDOWN: BASELINE vs AI HYBRID (Tight<0.40, Close<0.25)")
    print("=" * 102)
    for strat in ["MPS", "AIP", "HTF_ASIA"]:
        b_arr = base_df[base_df["strategy"] == strat]["r"].values
        h_arr = best_df[best_df["strategy"] == strat]["r"].values
        nb, wrb, arb, nrb, pfb, mddb, rmb = calc_stats(b_arr)
        nh, wrh, arh, nrh, pfh, mddh, rmh = calc_stats(h_arr)
        print(f"{strat:10s} | BASE: {nb:4d} tr, WR {wrb:5.1f}%, Net {nrb:+7.2f}R, PF {pfb:4.2f}, MDD {mddb:5.1f}R")
        print(f"{'':10s} | AI  : {nh:4d} tr, WR {wrh:5.1f}%, Net {nrh:+7.2f}R, PF {pfh:4.2f}, MDD {mddh:5.1f}R")
        print("-" * 102)


if __name__ == "__main__":
    main()