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
# AIP v10.13 Exact Parameters (Set 2: MACD Rule-Only)
# ==========================================
M1_PATH = "XAUUSD_M1.csv"
M5_PATH = "XAUUSD_M5.csv"

SL_ATR_MULT = 1.5
RR1 = 1.0
RR2 = 2.5
BE_BUFFER = 0.5
SPREAD = 0.25
MAX_HOLD_MIN = 240
FRI_NO_ENTRY_MIN = 1200
FRI_CLOSE_MIN = 1305
MIN_ADX = 18.0
N_FOLDS = 5


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Wilder's ADX(14) matching MT4 iADX."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs())
    )

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def load_data():
    """Load M1 and M5 CSV files and compute AIP v10.13 indicators."""
    print("Loading M1 and M5 CSV files...")
    m1 = pd.read_csv(
        M1_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m5 = pd.read_csv(
        M5_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    # H1 Trend
    h1 = m5.set_index("time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    h1["h1_ema50"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_ema200"] = h1["close"].ewm(span=200, adjust=False).mean()
    h1["h1_ready_time"] = h1["time"] + pd.Timedelta(hours=1)
    h1_r = h1[["h1_ready_time", "close", "h1_ema50", "h1_ema200"]].rename(columns={"close": "h1_close"})

    # M5 Indicators
    m5["adx14"] = compute_adx(m5, 14)
    ema12 = m5["close"].ewm(span=12, adjust=False).mean()
    ema26 = m5["close"].ewm(span=26, adjust=False).mean()
    m5["macd"] = ema12 - ema26
    m5["macd_sig"] = m5["macd"].rolling(9).mean()
    m5["macd_hist"] = m5["macd"] - m5["macd_sig"]

    prev_c = m5["close"].shift(1)
    tr = np.maximum(m5["high"] - m5["low"], np.maximum((m5["high"] - prev_c).abs(), (m5["low"] - prev_c).abs()))
    m5["atr14"] = tr.rolling(14).mean()

    delta = m5["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    m5["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    rng = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["body_ratio"] = (m5["close"] - m5["open"]) / rng
    m5["hist_norm"] = m5["macd_hist"] / m5["atr14"]

    m5["entry_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["dayofweek"] = m5["entry_time"].dt.dayofweek
    m5["min_of_day"] = m5["entry_time"].dt.hour * 60 + m5["entry_time"].dt.minute

    m5 = pd.merge_asof(m5, h1_r, left_on="entry_time", right_on="h1_ready_time", direction="backward")
    m5 = m5[(m5["entry_time"] >= m1["time"].min()) & (m5["entry_time"] <= m1["time"].max())].dropna().reset_index(drop=True)

    # Set 2: MACD Rule-Only signals
    fri_ok = ~((m5["dayofweek"] == 4) & (m5["min_of_day"] >= FRI_NO_ENTRY_MIN))
    adx_ok = m5["adx14"] >= MIN_ADX
    h1_up = (m5["h1_close"] > m5["h1_ema50"]) & (m5["h1_ema50"] > m5["h1_ema200"])
    h1_dn = (m5["h1_close"] < m5["h1_ema50"]) & (m5["h1_ema50"] < m5["h1_ema200"])

    m5["macd_buy"] = (fri_ok & adx_ok & h1_up & (m5["macd"] > m5["macd_sig"]) & (m5["macd"].shift(1) <= m5["macd_sig"].shift(1)) & (m5["macd"] < 0)).astype("int8")
    m5["macd_sell"] = (fri_ok & adx_ok & h1_dn & (m5["macd"] < m5["macd_sig"]) & (m5["macd"].shift(1) >= m5["macd_sig"].shift(1)) & (m5["macd"] > 0)).astype("int8")

    del h1, h1_r
    gc.collect()
    return m1, m5


def extract_trade_snapshots(m1_arrays, m5_lookup, sig_df):
    """
    Simulate each MACD trade bar-by-bar on M1 and record a snapshot at every closed M5 bar
    while the trade is still open. Used to train the Exit Manager AI.
    """
    m1_times, m1_opens, m1_highs, m1_lows, m1_closes, m1_dow, m1_mod = m1_arrays
    m5_times, m5_adx, m5_rsi, m5_hist, m5_body = m5_lookup
    n_m1 = len(m1_times)

    snapshots = []
    trade_meta = []

    for tid, r in sig_df.iterrows():
        entry_time = r["entry_time"].to_datetime64()
        direction = int(r["dir"])
        sl_dist = float(r["atr14"]) * SL_ATR_MULT
        if sl_dist <= 0:
            continue

        s_idx = np.searchsorted(m1_times, entry_time, side="left")
        if s_idx >= n_m1 - 1:
            continue

        max_exit_time = entry_time + np.timedelta64(MAX_HOLD_MIN, "m")
        entry = (m1_opens[s_idx] + SPREAD) if direction == 1 else m1_opens[s_idx]
        sl_p = entry - direction * sl_dist
        tp1_p = entry + direction * sl_dist * RR1
        tp2_p = entry + direction * sl_dist * RR2
        o2_sl = sl_p

        hit_tp1 = False
        mfe_r = 0.0
        mae_r = 0.0
        final_r = np.nan
        exit_time = pd.NaT
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
                            final_r, exit_time = RR1 + RR2, t_now
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        final_r, exit_time = RR1 + (o2_sl - entry) / sl_dist, t_now
                        break
                    if m1_highs[j] >= tp2_p:
                        final_r, exit_time = RR1 + RR2, t_now
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
                            final_r, exit_time = RR1 + RR2, t_now
                            break
                else:
                    if ah >= o2_sl:
                        final_r, exit_time = RR1 + (entry - o2_sl) / sl_dist, t_now
                        break
                    if al <= tp2_p:
                        final_r, exit_time = RR1 + RR2, t_now
                        break

            mfe_r = max(mfe_r, fav_r)
            mae_r = min(mae_r, adv_r)

            # Force-close on Friday 21:45 or 4h MaxHold
            if (m1_dow[j] == 4 and m1_mod[j] >= FRI_CLOSE_MIN) or (t_now >= max_exit_time):
                leg_r = direction * (cur_px - entry) / sl_dist
                final_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                exit_time = t_now
                break

            # Record snapshot at each closed M5 boundary (minute % 5 == 4, i.e., end of M5 bar)
            if m1_mod[j] % 5 == 4:
                next_m5_t = t_now + np.timedelta64(1, "m")
                m5_idx = np.searchsorted(m5_times, next_m5_t, side="right") - 1
                if m5_idx >= 0:
                    bars_held = int((next_m5_t - entry_time) / np.timedelta64(5, "m"))
                    leg_r = direction * (cur_px - entry) / sl_dist
                    curr_total_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                    # Directional features so BUY and SELL share a single unified Exit AI model
                    rsi_dir = m5_rsi[m5_idx] if direction == 1 else (100.0 - m5_rsi[m5_idx])
                    hist_dir = direction * m5_hist[m5_idx]
                    body_dir = direction * m5_body[m5_idx]

                    trade_snaps.append({
                        "trade_id": tid,
                        "entry_time": r["entry_time"],
                        "snap_time": pd.Timestamp(next_m5_t),
                        "bars_held": bars_held,
                        "hit_tp1": int(hit_tp1),
                        "curr_r": leg_r,
                        "curr_total_r": curr_total_r,
                        "mfe_r": mfe_r,
                        "mae_r": mae_r,
                        "adx14": m5_adx[m5_idx],
                        "rsi_dir": rsi_dir,
                        "hist_dir": hist_dir,
                        "body_dir": body_dir
                    })

        if not np.isnan(final_r):
            trade_meta.append({
                "trade_id": tid,
                "entry_time": r["entry_time"],
                "dir": direction,
                "atr14": float(r["atr14"]),
                "base_r": final_r,
                "base_exit": pd.Timestamp(exit_time)
            })
            for s in trade_snaps:
                s["final_r"] = final_r
                # Target = 1 if holding to normal exit beats closing right now
                s["target_hold"] = 1 if final_r > s["curr_total_r"] else 0
                snapshots.append(s)

    return pd.DataFrame(trade_meta), pd.DataFrame(snapshots)


def simulate_with_exit_ai(m1_arrays, m5_lookup, oos_trades, fold_models, time_edges, mode="baseline", close_th=0.30, tighten_th=0.40):
    """
    Run realistic non-overlapping M1 simulation on OOS trades using the fold-specific Exit AI model.
    mode:
      - 'baseline': normal v10.13 exit
      - 'early_close': close open leg(s) at M5 close if P(Hold) < close_th
      - 'hybrid': tighten SL if P(Hold) < tighten_th, and close if P(Hold) < close_th
    """
    m1_times, m1_opens, m1_highs, m1_lows, m1_closes, m1_dow, m1_mod = m1_arrays
    m5_times, m5_adx, m5_rsi, m5_hist, m5_body = m5_lookup
    n_m1 = len(m1_times)

    results = []
    next_free = pd.Timestamp.min

    for _, r in oos_trades.sort_values("entry_time").iterrows():
        if r["entry_time"] < next_free:
            continue

        if mode == "baseline":
            results.append(float(r["base_r"]))
            next_free = r["base_exit"]
            continue

        # Identify which fold model to use (strictly out-of-sample)
        fold_idx = int(r["fold"])
        model = fold_models.get(fold_idx, None)
        if model is None:
            results.append(float(r["base_r"]))
            next_free = r["base_exit"]
            continue

        entry_time = r["entry_time"].to_datetime64()
        direction = int(r["dir"])
        sl_dist = float(r["atr14"]) * SL_ATR_MULT
        s_idx = np.searchsorted(m1_times, entry_time, side="left")
        max_exit_time = entry_time + np.timedelta64(MAX_HOLD_MIN, "m")

        entry = (m1_opens[s_idx] + SPREAD) if direction == 1 else m1_opens[s_idx]
        sl_p = entry - direction * sl_dist
        tp1_p = entry + direction * sl_dist * RR1
        tp2_p = entry + direction * sl_dist * RR2
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
                        final_r = 2.0 * (o1_sl - entry) / sl_dist
                        exit_time = pd.Timestamp(t_now)
                        break
                    if m1_highs[j] >= tp1_p:
                        hit_tp1 = True
                        o2_sl = max(o2_sl, entry + BE_BUFFER)
                        if m1_highs[j] >= tp2_p:
                            final_r, exit_time = RR1 + RR2, pd.Timestamp(t_now)
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        final_r = RR1 + (o2_sl - entry) / sl_dist
                        exit_time = pd.Timestamp(t_now)
                        break
                    if m1_highs[j] >= tp2_p:
                        final_r, exit_time = RR1 + RR2, pd.Timestamp(t_now)
                        break
            else:
                ah, al = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                fav_r = (entry - al) / sl_dist
                adv_r = (entry - ah) / sl_dist
                cur_px = m1_closes[j] + SPREAD
                if not hit_tp1:
                    if ah >= o1_sl:
                        final_r = 2.0 * (entry - o1_sl) / sl_dist
                        exit_time = pd.Timestamp(t_now)
                        break
                    if al <= tp1_p:
                        hit_tp1 = True
                        o2_sl = min(o2_sl, entry - BE_BUFFER)
                        if al <= tp2_p:
                            final_r, exit_time = RR1 + RR2, pd.Timestamp(t_now)
                            break
                else:
                    if ah >= o2_sl:
                        final_r = RR1 + (entry - o2_sl) / sl_dist
                        exit_time = pd.Timestamp(t_now)
                        break
                    if al <= tp2_p:
                        final_r, exit_time = RR1 + RR2, pd.Timestamp(t_now)
                        break

            mfe_r = max(mfe_r, fav_r)
            mae_r = min(mae_r, adv_r)

            if (m1_dow[j] == 4 and m1_mod[j] >= FRI_CLOSE_MIN) or (t_now >= max_exit_time):
                leg_r = direction * (cur_px - entry) / sl_dist
                final_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                exit_time = pd.Timestamp(t_now)
                break

            # Evaluate Exit AI at each closed M5 bar
            if m1_mod[j] % 5 == 4:
                next_m5_t = t_now + np.timedelta64(1, "m")
                m5_idx = np.searchsorted(m5_times, next_m5_t, side="right") - 1
                if m5_idx >= 0:
                    bars_held = int((next_m5_t - entry_time) / np.timedelta64(5, "m"))
                    leg_r = direction * (cur_px - entry) / sl_dist
                    rsi_dir = m5_rsi[m5_idx] if direction == 1 else (100.0 - m5_rsi[m5_idx])
                    hist_dir = direction * m5_hist[m5_idx]
                    body_dir = direction * m5_body[m5_idx]

                    X_snap = np.array([[
                        bars_held, int(hit_tp1), leg_r, mfe_r, mae_r,
                        m5_adx[m5_idx], rsi_dir, hist_dir, body_dir
                    ]], dtype=np.float32)
                    p_hold = model.predict_proba(X_snap)[0, 1]

                    if p_hold < close_th:
                        # AI Early Close
                        final_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                        exit_time = pd.Timestamp(t_now)
                        break
                    elif mode == "hybrid" and p_hold < tighten_th:
                        # AI Tighten SL: cut risk to 0.5R if in loss, or move to BE if in profit
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

        results.append(final_r)
        next_free = exit_time

    arr = np.array(results, dtype=np.float64)
    net_r = arr.sum()
    wr = (arr > 0).mean() * 100.0
    avg_r = arr.mean()
    gp, gl = arr[arr > 0].sum(), abs(arr[arr < 0].sum())
    pf = gp / gl if gl > 0 else 9.99
    eq = np.concatenate([[0.0], np.cumsum(arr)])
    mdd = (np.maximum.accumulate(eq) - eq).max()
    return len(arr), wr, avg_r, net_r, pf, mdd, (net_r / mdd if mdd > 0 else 9.99)


def main():
    m1, m5 = load_data()
    m1_arrays = (
        m1["time"].values, m1["open"].values, m1["high"].values,
        m1["low"].values, m1["close"].values,
        m1["time"].dt.dayofweek.values, (m1["time"].dt.hour * 60 + m1["time"].dt.minute).values
    )
    m5_lookup = (
        m5["entry_time"].values, m5["adx14"].values, m5["rsi14"].values,
        m5["hist_norm"].values, m5["body_ratio"].values
    )

    sig_df = m5[(m5["macd_buy"] == 1) | (m5["macd_sell"] == 1)].copy().reset_index(drop=True)
    sig_df["dir"] = np.where(sig_df["macd_buy"] == 1, 1, -1)

    print(f"Extracting in-trade M5 snapshots for {len(sig_df):,} MACD setups...")
    trades_df, snaps_df = extract_trade_snapshots(m1_arrays, m5_lookup, sig_df)
    print(f"Extracted {len(snaps_df):,} M5 snapshots from {len(trades_df):,} completed trades.")

    t_min, t_max = m5["entry_time"].min(), m5["entry_time"].max() + pd.Timedelta(minutes=1)
    time_edges = pd.date_range(start=t_min, end=t_max, periods=N_FOLDS + 2)

    feat_cols = ["bars_held", "hit_tp1", "curr_r", "mfe_r", "mae_r", "adx14", "rsi_dir", "hist_dir", "body_dir"]

    # Train 5-Fold Expanding Walk-Forward Exit AI models
    print("Training 5-Fold Walk-Forward Exit Manager AI models...")
    fold_models = {}
    oos_trade_list = []

    for fold in range(1, N_FOLDS + 1):
        tr_end, te_start, te_end = time_edges[fold], time_edges[fold], time_edges[fold + 1]
        tr_snaps = snaps_df[snaps_df["snap_time"] < tr_end]
        te_trades = trades_df[(trades_df["entry_time"] >= te_start) & (trades_df["entry_time"] < te_end)].copy()
        te_trades["fold"] = fold
        oos_trade_list.append(te_trades)

        if len(tr_snaps) >= 200 and tr_snaps["target_hold"].nunique() == 2:
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=120, learning_rate=0.03,
                max_depth=3, num_leaves=7, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, n_jobs=2, random_state=42, verbose=-1
            )
            model.fit(tr_snaps[feat_cols].values, tr_snaps["target_hold"].values)
            fold_models[fold] = model

    oos_trades = pd.concat(oos_trade_list).sort_values("entry_time").reset_index(drop=True)

    print("\n" + "=" * 102)
    print("5-FOLD OOS: EXIT MANAGER AI vs BASELINE (SET 2: MACD RULE-ONLY)")
    print("=" * 102)
    header = f"{'Exit Policy':34s} | {'Trades':>6s} | {'WinRate':>7s} | {'Avg R':>7s} | {'Net R':>8s} | {'PF':>5s} | {'MDD(R)':>7s} | {'Net/MDD':>7s}"
    print(header)
    print("-" * len(header))

    configs = [
        ("Baseline (Normal v10.13 Exit)", "baseline", 0.0, 0.0),
        ("AI Early Close (P_hold < 0.25)", "early_close", 0.25, 0.0),
        ("AI Early Close (P_hold < 0.30)", "early_close", 0.30, 0.0),
        ("AI Early Close (P_hold < 0.35)", "early_close", 0.35, 0.0),
        ("AI Early Close (P_hold < 0.40)", "early_close", 0.40, 0.0),
        ("AI Hybrid (Tight<0.40, Close<0.25)", "hybrid", 0.25, 0.40),
    ]

    for label, mode, c_th, t_th in configs:
        n, wr, ar, nr, pf, mdd, rm = simulate_with_exit_ai(
            m1_arrays, m5_lookup, oos_trades, fold_models, time_edges,
            mode=mode, close_th=c_th, tighten_th=t_th
        )
        print(f"{label:34s} | {n:6d} | {wr:6.2f}% | {ar:+6.3f}R | {nr:+7.2f}R | {pf:5.2f} | {mdd:6.1f}R | {rm:+7.2f}")


if __name__ == "__main__":
    main()