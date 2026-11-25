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
# Configuration Parameters
# ==========================================
M1_PATH = "XAUUSD_M1.csv"
M5_PATH = "XAUUSD_M5.csv"

SL_ATR_MULT = 1.5       # SL distance = M5 ATR(14) * SL_ATR_MULT
RR1 = 1.0               # TP1 Risk-Reward ratio
RR2 = 2.5               # TP2 Risk-Reward ratio (must be > RR1)
MOVE_BE_ON_TP1 = True   # Move Order 2 SL to Breakeven after TP1 is hit
SPREAD = 0.25           # Spread in price units (e.g., 0.25 = 25 points)
MAX_HOLD_M1_BARS = 240  # Max 240 M1 bars (4 hours) to wait for TP2/SL
N_FOLDS = 5             # 5-Fold Expanding Walk-Forward Validation

MIN_M5_ADX = 18.0
EXCLUDED_HOURS = {0, 9, 14, 19, 21}


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 columns to float32 to reduce RAM usage by 50%."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Wilder's ADX indicator."""
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


def load_data(m1_path: str, m5_path: str):
    """Load M1 (execution) and M5 (signals + H1 trend) without look-ahead bias."""
    print("Loading M1 and M5 CSV files...")
    m1 = pd.read_csv(
        m1_path, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m5 = pd.read_csv(
        m5_path, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m1, m5 = downcast_floats(m1), downcast_floats(m5)

    # 1. Resample M5 to H1 for Major Trend Filter
    h1 = m5.set_index("time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    h1["h1_ema50"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_ema200"] = h1["close"].ewm(span=200, adjust=False).mean()
    h1["h1_ready_time"] = h1["time"] + pd.Timedelta(hours=1)
    h1_renamed = h1[["h1_ready_time", "close", "h1_ema50", "h1_ema200"]].rename(
        columns={"close": "h1_close"}
    )

    # 2. Compute M5 indicators
    m5["m5_ema9"] = m5["close"].ewm(span=9, adjust=False).mean()
    m5["m5_ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["m5_ema50"] = m5["close"].ewm(span=50, adjust=False).mean()
    m5["m5_adx14"] = compute_adx(m5, period=14)

    # MACD (12, 26, 9)
    ema12 = m5["close"].ewm(span=12, adjust=False).mean()
    ema26 = m5["close"].ewm(span=26, adjust=False).mean()
    m5["m5_macd"] = ema12 - ema26
    m5["m5_macd_sig"] = m5["m5_macd"].ewm(span=9, adjust=False).mean()

    prev_close = m5["close"].shift(1)
    tr = np.maximum(
        m5["high"] - m5["low"],
        np.maximum((m5["high"] - prev_close).abs(), (m5["low"] - prev_close).abs())
    )
    m5["m5_atr14"] = tr.rolling(14).mean()
    m5["m5_atr50"] = tr.rolling(50).mean()

    # M5 RSI(14)
    delta = m5["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    m5["m5_rsi14"] = 100 - (100 / (1 + rs))

    # M5 candle anatomy & normalized features
    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["m5_body_ratio"] = (m5["close"] - m5["open"]) / candle_range
    m5["m5_upper_wick"] = (m5["high"] - np.maximum(m5["open"], m5["close"])) / candle_range
    m5["m5_lower_wick"] = (np.minimum(m5["open"], m5["close"]) - m5["low"]) / candle_range
    m5["m5_range_atr"] = candle_range / m5["m5_atr14"]
    m5["m5_vol_ratio"] = m5["m5_atr14"] / m5["m5_atr50"]
    m5["m5_trend_gap"] = (m5["m5_ema20"] - m5["m5_ema50"]) / m5["m5_atr14"]
    m5["m5_dist_ema50"] = (m5["close"] - m5["m5_ema50"]) / m5["m5_atr14"]
    m5["m5_macd_norm"] = m5["m5_macd"] / m5["m5_atr14"]

    # Entry time = M5 candle close (+5 minutes)
    m5["entry_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["hour"] = m5["entry_time"].dt.hour.astype("int8")
    m5["dayofweek"] = m5["entry_time"].dt.dayofweek.astype("int8")

    # Merge closed H1 onto M5
    m5 = pd.merge_asof(
        m5, h1_renamed, left_on="entry_time", right_on="h1_ready_time", direction="backward"
    )
    m5 = m5[(m5["entry_time"] >= m1["time"].min()) & (m5["entry_time"] <= m1["time"].max())].copy()

    # Common HTF & ADX trend filters
    adx_ok = m5["m5_adx14"] >= MIN_M5_ADX
    h1_up = (m5["h1_close"] > m5["h1_ema50"]) & (m5["h1_ema50"] > m5["h1_ema200"])
    h1_dn = (m5["h1_close"] < m5["h1_ema50"]) & (m5["h1_ema50"] < m5["h1_ema200"])

    # --- Setup 1 (S6): Inside Bar Breakout ---
    ib = (m5["high"].shift(1) < m5["high"].shift(2)) & (m5["low"].shift(1) > m5["low"].shift(2))
    ib_buy = h1_up & adx_ok & ib & (m5["close"] > m5["high"].shift(2))
    ib_sell = h1_dn & adx_ok & ib & (m5["close"] < m5["low"].shift(2))

    # --- Setup 2 (S2): MACD Zero-Line Pullback Cross ---
    macd_buy = (
        h1_up & adx_ok
        & (m5["m5_macd"] > m5["m5_macd_sig"])
        & (m5["m5_macd"].shift(1) <= m5["m5_macd_sig"].shift(1))
        & (m5["m5_macd"] < 0)
    )
    macd_sell = (
        h1_dn & adx_ok
        & (m5["m5_macd"] < m5["m5_macd_sig"])
        & (m5["m5_macd"].shift(1) >= m5["m5_macd_sig"].shift(1))
        & (m5["m5_macd"] > 0)
    )

    # --- Setup 3 (S3): EMA 9/20 Crossover ---
    ema_buy = (
        h1_up & adx_ok
        & (m5["m5_ema9"] > m5["m5_ema20"])
        & (m5["m5_ema9"].shift(1) <= m5["m5_ema20"].shift(1))
    )
    ema_sell = (
        h1_dn & adx_ok
        & (m5["m5_ema9"] < m5["m5_ema20"])
        & (m5["m5_ema9"].shift(1) >= m5["m5_ema20"].shift(1))
    )

    # Store setup flags as features for AI
    m5["is_ib"] = (ib_buy | ib_sell).astype("int8")
    m5["is_macd"] = (macd_buy | macd_sell).astype("int8")
    m5["is_ema_cross"] = (ema_buy | ema_sell).astype("int8")

    m5["setup_buy"] = (ib_buy | macd_buy | ema_buy).astype("int8")
    m5["setup_sell"] = (ib_sell | macd_sell | ema_sell).astype("int8")

    # Tag primary setup name for breakdown reporting
    m5["setup_tag"] = np.where(
        m5["is_ib"] == 1, "S6_InsideBar",
        np.where(m5["is_macd"] == 1, "S2_MACD", "S3_EMA9_20")
    )

    m5 = m5.dropna().reset_index(drop=True)
    del h1, h1_renamed
    gc.collect()
    return m1, downcast_floats(m5)


def simulate_m5_setups_on_m1(m5: pd.DataFrame, m1: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Simulate exact 2-order execution bar-by-bar on M1 data for each M5 signal."""
    setup_col = "setup_buy" if direction == "BUY" else "setup_sell"
    sig_df = m5[m5[setup_col] == 1].copy().reset_index(drop=True)

    m1_times = m1["time"].values
    m1_opens = m1["open"].values
    m1_highs = m1["high"].values
    m1_lows = m1["low"].values
    n_m1 = len(m1)

    r0 = -2.0
    r1 = RR1 if MOVE_BE_ON_TP1 else (RR1 - 1.0)
    r2 = RR1 + RR2

    outcomes = np.full(len(sig_df), np.nan, dtype=np.float32)
    exit_times = np.full(len(sig_df), np.datetime64("NaT"), dtype="datetime64[ns]")

    entry_times = sig_df["entry_time"].values
    atrs = sig_df["m5_atr14"].values

    for i in range(len(sig_df)):
        start_idx = np.searchsorted(m1_times, entry_times[i], side="left")
        if start_idx >= n_m1 - MAX_HOLD_M1_BARS:
            continue

        sl_dist = atrs[i] * SL_ATR_MULT
        if sl_dist <= 0:
            continue

        hit_tp1 = False
        end_idx = min(start_idx + MAX_HOLD_M1_BARS, n_m1)

        if direction == "BUY":
            entry = m1_opens[start_idx] + SPREAD
            sl_p, tp1_p, tp2_p = entry - sl_dist, entry + sl_dist * RR1, entry + sl_dist * RR2
            o2_sl = sl_p
            for j in range(start_idx, end_idx):
                if not hit_tp1:
                    if m1_lows[j] <= sl_p:
                        outcomes[i], exit_times[i] = r0, m1_times[j]
                        break
                    if m1_highs[j] >= tp1_p:
                        hit_tp1 = True
                        if MOVE_BE_ON_TP1:
                            o2_sl = entry
                        if m1_highs[j] >= tp2_p:
                            outcomes[i], exit_times[i] = r2, m1_times[j]
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        outcomes[i], exit_times[i] = r1, m1_times[j]
                        break
                    if m1_highs[j] >= tp2_p:
                        outcomes[i], exit_times[i] = r2, m1_times[j]
                        break
        else:
            entry = m1_opens[start_idx]
            sl_p, tp1_p, tp2_p = entry + sl_dist, entry - sl_dist * RR1, entry - sl_dist * RR2
            o2_sl = sl_p
            for j in range(start_idx, end_idx):
                ask_h, ask_l = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                if not hit_tp1:
                    if ask_h >= sl_p:
                        outcomes[i], exit_times[i] = r0, m1_times[j]
                        break
                    if ask_l <= tp1_p:
                        hit_tp1 = True
                        if MOVE_BE_ON_TP1:
                            o2_sl = entry
                        if ask_l <= tp2_p:
                            outcomes[i], exit_times[i] = r2, m1_times[j]
                            break
                else:
                    if ask_h >= o2_sl:
                        outcomes[i], exit_times[i] = r1, m1_times[j]
                        break
                    if ask_l <= tp2_p:
                        outcomes[i], exit_times[i] = r2, m1_times[j]
                        break

    sig_df["outcome_r"] = outcomes
    sig_df["exit_time"] = exit_times
    sig_df["direction"] = direction
    sig_df = sig_df.dropna(subset=["outcome_r"]).reset_index(drop=True)
    sig_df["target_win"] = (sig_df["outcome_r"] > 0).astype("int8")
    return sig_df


def build_lgb_model():
    """Create a regularized binary LightGBM classifier."""
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=42,
        verbose=-1
    )


def compute_metrics(df_sub: pd.DataFrame, use_ai: bool = False, prob_th: float = 0.50, filter_hours: bool = False) -> dict:
    """Run non-overlapping simulation and compute Net R, WinRate, Profit Factor, and Max Drawdown."""
    trades = []
    next_free_time = pd.Timestamp.min

    for _, row in df_sub.sort_values("entry_time").iterrows():
        if row["entry_time"] < next_free_time:
            continue
        if filter_hours and int(row["hour"]) in EXCLUDED_HOURS:
            continue
        if use_ai and row["prob_win"] < prob_th:
            continue

        trades.append(float(row["outcome_r"]))
        next_free_time = row["exit_time"]

    if len(trades) == 0:
        return {"trades": 0, "wr": 0.0, "avg_r": 0.0, "net_r": 0.0, "pf": 0.0, "mdd_r": 0.0, "ret_mdd": 0.0}

    arr = np.array(trades, dtype=np.float64)
    net_r = arr.sum()
    wr = (arr > 0).mean() * 100.0
    avg_r = arr.mean()

    gross_profit = arr[arr > 0].sum()
    gross_loss = abs(arr[arr < 0].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.concatenate([[0.0], np.cumsum(arr)])
    running_peak = np.maximum.accumulate(equity)
    mdd_r = (running_peak - equity).max()
    ret_mdd = (net_r / mdd_r) if mdd_r > 0 else float("inf")

    return {"trades": len(arr), "wr": wr, "avg_r": avg_r, "net_r": net_r, "pf": pf, "mdd_r": mdd_r, "ret_mdd": ret_mdd}


def run_walk_forward(buy_df: pd.DataFrame, sell_df: pd.DataFrame, feature_cols: list, time_edges: list):
    """Run N-Fold expanding-window walk-forward validation."""
    oos_frames = []

    print("\n" + "=" * 90)
    print(f"RUNNING {N_FOLDS}-FOLD WALK-FORWARD: MULTI-SETUP (InsideBar + MACD + EMA9/20)")
    print("=" * 90)

    for fold in range(1, N_FOLDS + 1):
        train_end = time_edges[fold]
        test_start = time_edges[fold]
        test_end = time_edges[fold + 1]

        fold_oos_list = []
        for _, df_dir in [("BUY", buy_df), ("SELL", sell_df)]:
            tr = df_dir[df_dir["entry_time"] < train_end]
            te = df_dir[(df_dir["entry_time"] >= test_start) & (df_dir["entry_time"] < test_end)].copy()

            if len(te) == 0:
                continue

            if len(tr) >= 50 and tr["target_win"].nunique() == 2:
                model = build_lgb_model()
                model.fit(tr[feature_cols], tr["target_win"])
                te["prob_win"] = model.predict_proba(te[feature_cols])[:, 1]
            else:
                te["prob_win"] = 0.0

            te["fold"] = fold
            fold_oos_list.append(te)

        if not fold_oos_list:
            continue

        fold_df = pd.concat(fold_oos_list).sort_values("entry_time")
        oos_frames.append(fold_df)

        m_base = compute_metrics(fold_df, use_ai=False, filter_hours=False)
        m_ai = compute_metrics(fold_df, use_ai=True, prob_th=0.52, filter_hours=False)

        print(f"Fold {fold} ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}):")
        print(f"  Base Only         | Tr: {m_base['trades']:3d} | WR: {m_base['wr']:5.1f}% | Net: {m_base['net_r']:+6.1f}R | PF: {m_base['pf']:4.2f} | MDD: {m_base['mdd_r']:5.1f}R")
        print(f"  AI (Prob >= 0.52) | Tr: {m_ai['trades']:3d} | WR: {m_ai['wr']:5.1f}% | Net: {m_ai['net_r']:+6.1f}R | PF: {m_ai['pf']:4.2f} | MDD: {m_ai['mdd_r']:5.1f}R")

    return pd.concat(oos_frames).sort_values("entry_time").reset_index(drop=True)


def print_summary_table(title: str, df_oos: pd.DataFrame, filter_hours: bool):
    print("\n" + "=" * 95)
    print(f"=== {title} ===")
    print("=" * 95)
    header = f"{'Strategy':24s} | {'Trades':>6s} | {'WinRate':>7s} | {'Avg R':>7s} | {'Net R':>8s} | {'PF':>5s} | {'MDD(R)':>7s} | {'Net/MDD':>7s}"
    print(header)
    print("-" * len(header))

    m = compute_metrics(df_oos, use_ai=False, filter_hours=filter_hours)
    print(f"{'Base Setup Only (No AI)':24s} | {m['trades']:6d} | {m['wr']:6.2f}% | {m['avg_r']:+6.3f}R | {m['net_r']:+7.2f}R | {m['pf']:5.2f} | {m['mdd_r']:6.1f}R | {m['ret_mdd']:+7.2f}")

    for th in [0.50, 0.52, 0.55, 0.58]:
        m = compute_metrics(df_oos, use_ai=True, prob_th=th, filter_hours=filter_hours)
        label = f"AI Filter (Prob >= {th:.2f})"
        print(f"{label:24s} | {m['trades']:6d} | {m['wr']:6.2f}% | {m['avg_r']:+6.3f}R | {m['net_r']:+7.2f}R | {m['pf']:5.2f} | {m['mdd_r']:6.1f}R | {m['ret_mdd']:+7.2f}")


def main():
    m1, m5 = load_data(M1_PATH, M5_PATH)

    print("Simulating Multi-Setup (InsideBar + MACD + EMA9/20) on M1 execution path...")
    buy_df = simulate_m5_setups_on_m1(m5, m1, "BUY")
    sell_df = simulate_m5_setups_on_m1(m5, m1, "SELL")

    print(f"Total Valid Setups -> BUY: {len(buy_df):,} | SELL: {len(sell_df):,}")

    t_min = m5["entry_time"].min()
    t_max = m5["entry_time"].max() + pd.Timedelta(minutes=1)
    time_edges = pd.date_range(start=t_min, end=t_max, periods=N_FOLDS + 2)

    del m1, m5
    gc.collect()

    feature_cols = [
        "m5_body_ratio", "m5_upper_wick", "m5_lower_wick",
        "m5_range_atr", "m5_vol_ratio", "m5_trend_gap",
        "m5_dist_ema50", "m5_macd_norm", "m5_adx14", "m5_rsi14",
        "is_ib", "is_macd", "is_ema_cross",
        "hour", "dayofweek"
    ]

    all_oos = run_walk_forward(buy_df, sell_df, feature_cols, time_edges)

    print_summary_table("MULTI-SETUP COMBINED | ALL HOURS", all_oos, filter_hours=False)
    print_summary_table("MULTI-SETUP COMBINED | EXCLUDE TOXIC HOURS", all_oos, filter_hours=True)

    print("\n=== BREAKDOWN BY SETUP TAG (OOS, AI Prob >= 0.52, All Hours) ===")
    for tag in ["S6_InsideBar", "S2_MACD", "S3_EMA9_20"]:
        sub_tag = all_oos[all_oos["setup_tag"] == tag]
        m = compute_metrics(sub_tag, use_ai=True, prob_th=0.52, filter_hours=False)
        print(f"{tag:14s} | Trades: {m['trades']:4d} | WR: {m['wr']:5.2f}% | Avg R: {m['avg_r']:+.3f}R | Net R: {m['net_r']:+7.2f}R | PF: {m['pf']:4.2f} | MDD: {m['mdd_r']:5.1f}R")


if __name__ == "__main__":
    main()