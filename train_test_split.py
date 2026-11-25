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


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 columns to float32 to reduce RAM usage by 50%."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def load_data(m1_path: str, m5_path: str):
    """Load M1 (for execution) and M5 (for signals + H1 trend) without look-ahead bias."""
    print("Loading M1 and M5 CSV files...")
    m1 = pd.read_csv(
        m1_path,
        usecols=["time", "open", "high", "low", "close"],
        parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m5 = pd.read_csv(
        m5_path,
        usecols=["time", "open", "high", "low", "close"],
        parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m1 = downcast_floats(m1)
    m5 = downcast_floats(m5)

    # 1. Resample M5 to H1 for Major Trend Filter
    h1 = m5.set_index("time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    h1["h1_ema50"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_ema200"] = h1["close"].ewm(span=200, adjust=False).mean()

    # Shift H1 by +1 hour so M5 only sees closed H1 candles
    h1["h1_ready_time"] = h1["time"] + pd.Timedelta(hours=1)
    h1_renamed = h1[["h1_ready_time", "close", "h1_ema50", "h1_ema200"]].rename(
        columns={"close": "h1_close"}
    )

    # 2. Compute M5 indicators and features
    m5["m5_ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["m5_ema50"] = m5["close"].ewm(span=50, adjust=False).mean()

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

    # M5 candle anatomy
    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["m5_body_ratio"] = (m5["close"] - m5["open"]) / candle_range
    m5["m5_upper_wick"] = (m5["high"] - np.maximum(m5["open"], m5["close"])) / candle_range
    m5["m5_lower_wick"] = (np.minimum(m5["open"], m5["close"]) - m5["low"]) / candle_range
    m5["m5_range_atr"] = candle_range / m5["m5_atr14"]
    m5["m5_vol_ratio"] = m5["m5_atr14"] / m5["m5_atr50"]
    m5["m5_trend_gap"] = (m5["m5_ema20"] - m5["m5_ema50"]) / m5["m5_atr14"]

    # Entry time is when the M5 candle closes (+5 minutes)
    m5["entry_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["hour"] = m5["entry_time"].dt.hour.astype("int8")
    m5["dayofweek"] = m5["entry_time"].dt.dayofweek.astype("int8")

    # Merge closed H1 onto M5 at M5 close time
    m5 = pd.merge_asof(
        m5,
        h1_renamed,
        left_on="entry_time",
        right_on="h1_ready_time",
        direction="backward"
    )

    # Filter M5 to the overlapping period covered by M1 data
    m1_min_time = m1["time"].min()
    m1_max_time = m1["time"].max()
    m5 = m5[(m5["entry_time"] >= m1_min_time) & (m5["entry_time"] <= m1_max_time)].copy()

    # Define M5 Base Setups (aligned with H1 EMA50 + EMA200)
    h1_up = (m5["h1_close"] > m5["h1_ema50"]) & (m5["h1_ema50"] > m5["h1_ema200"])
    h1_dn = (m5["h1_close"] < m5["h1_ema50"]) & (m5["h1_ema50"] < m5["h1_ema200"])

    m5_up = (m5["close"] > m5["m5_ema20"]) & (m5["m5_ema20"] > m5["m5_ema50"])
    m5_dn = (m5["close"] < m5["m5_ema20"]) & (m5["m5_ema20"] < m5["m5_ema50"])

    # M5 Pullback to EMA20 + Reclaim
    buy_pb = (m5["low"] <= m5["m5_ema20"]) | (m5["low"].shift(1) <= m5["m5_ema20"].shift(1))
    buy_trig = (m5["close"] > m5["open"]) & (m5["close"] > m5["m5_ema20"])
    m5["setup_buy"] = (h1_up & m5_up & buy_pb & buy_trig).astype("int8")

    sell_pb = (m5["high"] >= m5["m5_ema20"]) | (m5["high"].shift(1) >= m5["m5_ema20"].shift(1))
    sell_trig = (m5["close"] < m5["open"]) & (m5["close"] < m5["m5_ema20"])
    m5["setup_sell"] = (h1_dn & m5_dn & sell_pb & sell_trig).astype("int8")

    m5 = m5.dropna().reset_index(drop=True)
    del h1, h1_renamed
    gc.collect()

    return m1, downcast_floats(m5)


def simulate_m5_setups_on_m1(m5: pd.DataFrame, m1: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    For each M5 signal, simulate exact 2-order execution bar-by-bar on M1 data.
    Returns a DataFrame of completed setups with outcome_r, binary target, and exit_time.
    """
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
        # Find the first M1 bar at or after M5 close time
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
            sl_p = entry - sl_dist
            tp1_p = entry + sl_dist * RR1
            tp2_p = entry + sl_dist * RR2
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
            sl_p = entry + sl_dist
            tp1_p = entry - sl_dist * RR1
            tp2_p = entry - sl_dist * RR2
            o2_sl = sl_p

            for j in range(start_idx, end_idx):
                ask_h = m1_highs[j] + SPREAD
                ask_l = m1_lows[j] + SPREAD
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

    # Keep only setups that finished within MAX_HOLD_M1_BARS
    sig_df = sig_df.dropna(subset=["outcome_r"]).reset_index(drop=True)
    # Binary target: 1 = Hit at least TP1 (> 0R), 0 = Hit SL (-2R)
    sig_df["target_win"] = (sig_df["outcome_r"] > 0).astype("int8")
    return sig_df


def train_binary_model(sig_df: pd.DataFrame, direction: str, feature_cols: list, split_time: pd.Timestamp):
    """Train regularized binary LightGBM to predict probability of hitting TP1 before SL."""
    train_df = sig_df[sig_df["entry_time"] < split_time]
    test_df = sig_df[sig_df["entry_time"] >= split_time].copy()

    print(f"\n--- {direction} M5 Model (Binary: TP1 vs SL) ---")
    print(f"Valid Setups: {len(sig_df):,} (Train: {len(train_df):,}, Test: {len(test_df):,})")
    print(f"Overall Win Rate (> 0R): {sig_df['target_win'].mean() * 100:.2f}%")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=42,
        verbose=-1
    )
    model.fit(train_df[feature_cols], train_df["target_win"])

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 5 Features:", imp.head(5).to_dict())

    test_df["prob_win"] = model.predict_proba(test_df[feature_cols])[:, 1]
    return test_df


def run_non_overlapping_backtest(df_sub: pd.DataFrame, use_ai: bool = False, prob_th: float = 0.50):
    """Simulate realistic execution: only 1 active 2-order set at a time."""
    trades = []
    next_free_time = pd.Timestamp.min

    for _, row in df_sub.sort_values("entry_time").iterrows():
        if row["entry_time"] < next_free_time:
            continue
        if use_ai and row["prob_win"] < prob_th:
            continue

        trades.append(row["outcome_r"])
        next_free_time = row["exit_time"]

    trades = np.array(trades)
    if len(trades) == 0:
        return 0, 0.0, 0.0, 0.0

    win_rate = (trades > 0).mean() * 100
    avg_r = trades.mean()
    net_r = trades.sum()
    return len(trades), win_rate, avg_r, net_r


def print_backtest_table(title: str, test_df: pd.DataFrame):
    print(f"\n=== {title} ===")
    n_tr, wr, avg_r, net_r = run_non_overlapping_backtest(test_df, use_ai=False)
    print(f"Base Setup Only (No AI) | Trades: {n_tr:4d} | WinRate: {wr:5.2f}% | Avg R: {avg_r:+.3f}R | Net R: {net_r:+.2f}R")

    for th in [0.45, 0.50, 0.52, 0.55, 0.58]:
        n_tr, wr, avg_r, net_r = run_non_overlapping_backtest(test_df, use_ai=True, prob_th=th)
        print(f"AI Filter (Prob >= {th:.2f}) | Trades: {n_tr:4d} | WinRate: {wr:5.2f}% | Avg R: {avg_r:+.3f}R | Net R: {net_r:+.2f}R")


def main():
    m1, m5 = load_data(M1_PATH, M5_PATH)

    print("Simulating M5 BUY setups on M1 execution path...")
    buy_df = simulate_m5_setups_on_m1(m5, m1, "BUY")

    print("Simulating M5 SELL setups on M1 execution path...")
    sell_df = simulate_m5_setups_on_m1(m5, m1, "SELL")

    # Chronological 70/30 split timestamp based on M5 overlapping data
    split_time = m5["entry_time"].iloc[int(len(m5) * 0.7)]
    print(f"\nChronological Train/Test Split Time: {split_time}")

    del m1, m5
    gc.collect()

    # Features focused on M5 candle structure, volatility, momentum, and session time
    feature_cols = [
        "m5_body_ratio", "m5_upper_wick", "m5_lower_wick",
        "m5_range_atr", "m5_vol_ratio", "m5_trend_gap",
        "m5_rsi14", "hour", "dayofweek"
    ]

    test_buy = train_binary_model(buy_df, "BUY", feature_cols, split_time)
    test_sell = train_binary_model(sell_df, "SELL", feature_cols, split_time)
    test_combined = pd.concat([test_buy, test_sell]).sort_values("entry_time")

    print_backtest_table("Out-of-Sample: M5 BUY Only", test_buy)
    print_backtest_table("Out-of-Sample: M5 SELL Only", test_sell)
    print_backtest_table("Out-of-Sample: M5 COMBINED (BUY + SELL)", test_combined)


if __name__ == "__main__":
    main()