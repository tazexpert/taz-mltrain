import os

# Limit OpenBLAS and OpenMP threads BEFORE importing numpy, pandas, or lightgbm
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import m2cgen as m2c

# ==========================================
# Configuration Parameters
# ==========================================
M1_PATH = "XAUUSD_M1.csv"
M5_PATH = "XAUUSD_M5.csv"
OUTPUT_MQH = "XAUUSD_M5_MACD_AI_Models.mqh"

SL_ATR_MULT = 1.5
RR1 = 1.0
RR2 = 2.5
MOVE_BE_ON_TP1 = True
SPREAD = 0.25
MAX_HOLD_M1_BARS = 240
MIN_M5_ADX = 18.0


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
    """Load M1 (execution) and M5 (S2_MACD signals + H1 trend) without look-ahead bias."""
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
    m5["m5_ema20"] = m5["close"].ewm(span=20, adjust=False).mean()
    m5["m5_ema50"] = m5["close"].ewm(span=50, adjust=False).mean()
    m5["m5_adx14"] = compute_adx(m5, period=14)

    # MACD (12, 26, 9)
    ema12 = m5["close"].ewm(span=12, adjust=False).mean()
    ema26 = m5["close"].ewm(span=26, adjust=False).mean()
    m5["m5_macd"] = ema12 - ema26
    m5["m5_macd_sig"] = m5["m5_macd"].ewm(span=9, adjust=False).mean()
    m5["m5_macd_hist"] = m5["m5_macd"] - m5["m5_macd_sig"]

    prev_close = m5["close"].shift(1)
    tr = np.maximum(
        m5["high"] - m5["low"],
        np.maximum((m5["high"] - prev_close).abs(), (m5["low"] - prev_close).abs())
    )
    m5["m5_atr14"] = tr.rolling(14).mean()
    m5["m5_atr50"] = tr.rolling(50).mean()

    delta = m5["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    m5["m5_rsi14"] = 100 - (100 / (1 + rs))

    candle_range = (m5["high"] - m5["low"]).replace(0, np.nan)
    m5["m5_body_ratio"] = (m5["close"] - m5["open"]) / candle_range
    m5["m5_upper_wick"] = (m5["high"] - np.maximum(m5["open"], m5["close"])) / candle_range
    m5["m5_lower_wick"] = (np.minimum(m5["open"], m5["close"]) - m5["low"]) / candle_range
    m5["m5_range_atr"] = candle_range / m5["m5_atr14"]
    m5["m5_vol_ratio"] = m5["m5_atr14"] / m5["m5_atr50"]
    m5["m5_trend_gap"] = (m5["m5_ema20"] - m5["m5_ema50"]) / m5["m5_atr14"]
    m5["m5_macd_norm"] = m5["m5_macd"] / m5["m5_atr14"]
    m5["m5_hist_norm"] = m5["m5_macd_hist"] / m5["m5_atr14"]

    m5["entry_time"] = m5["time"] + pd.Timedelta(minutes=5)
    m5["hour"] = m5["entry_time"].dt.hour.astype("int8")
    m5["dayofweek"] = m5["entry_time"].dt.dayofweek.astype("int8")

    m5 = pd.merge_asof(
        m5, h1_renamed, left_on="entry_time", right_on="h1_ready_time", direction="backward"
    )
    m5 = m5[(m5["entry_time"] >= m1["time"].min()) & (m5["entry_time"] <= m1["time"].max())].copy()

    # S2_MACD Base Setup ONLY
    adx_ok = m5["m5_adx14"] >= MIN_M5_ADX
    h1_up = (m5["h1_close"] > m5["h1_ema50"]) & (m5["h1_ema50"] > m5["h1_ema200"])
    h1_dn = (m5["h1_close"] < m5["h1_ema50"]) & (m5["h1_ema50"] < m5["h1_ema200"])

    m5["setup_buy"] = (
        h1_up & adx_ok
        & (m5["m5_macd"] > m5["m5_macd_sig"])
        & (m5["m5_macd"].shift(1) <= m5["m5_macd_sig"].shift(1))
        & (m5["m5_macd"] < 0)
    ).astype("int8")

    m5["setup_sell"] = (
        h1_dn & adx_ok
        & (m5["m5_macd"] < m5["m5_macd_sig"])
        & (m5["m5_macd"].shift(1) >= m5["m5_macd_sig"].shift(1))
        & (m5["m5_macd"] > 0)
    ).astype("int8")

    m5 = m5.dropna().reset_index(drop=True)
    del h1, h1_renamed
    gc.collect()
    return m1, downcast_floats(m5)


def simulate_m5_setups_on_m1(m5: pd.DataFrame, m1: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Label each S2_MACD setup using exact M1 execution path."""
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
                        outcomes[i] = r0
                        break
                    if m1_highs[j] >= tp1_p:
                        hit_tp1 = True
                        if MOVE_BE_ON_TP1:
                            o2_sl = entry
                        if m1_highs[j] >= tp2_p:
                            outcomes[i] = r2
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        outcomes[i] = r1
                        break
                    if m1_highs[j] >= tp2_p:
                        outcomes[i] = r2
                        break
        else:
            entry = m1_opens[start_idx]
            sl_p, tp1_p, tp2_p = entry + sl_dist, entry - sl_dist * RR1, entry - sl_dist * RR2
            o2_sl = sl_p
            for j in range(start_idx, end_idx):
                ask_h, ask_l = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                if not hit_tp1:
                    if ask_h >= sl_p:
                        outcomes[i] = r0
                        break
                    if ask_l <= tp1_p:
                        hit_tp1 = True
                        if MOVE_BE_ON_TP1:
                            o2_sl = entry
                        if ask_l <= tp2_p:
                            outcomes[i] = r2
                            break
                else:
                    if ask_h >= o2_sl:
                        outcomes[i] = r1
                        break
                    if ask_l <= tp2_p:
                        outcomes[i] = r2
                        break

    sig_df["outcome_r"] = outcomes
    sig_df = sig_df.dropna(subset=["outcome_r"]).reset_index(drop=True)
    sig_df["target_win"] = (sig_df["outcome_r"] > 0).astype("int8")
    return sig_df


def convert_c_to_mql4(c_code: str, func_name: str) -> str:
    """Convert m2cgen C output into clean, compilable MQL4 code."""
    code = re.sub(r"#include\s*<[^>]+>\s*", "", c_code)
    code = code.replace("double * input", "const double &input[]")
    code = re.sub(r"\bdouble\s+score\s*\(", f"double {func_name}(", code)
    code = re.sub(r"\bexp\s*\(", "MathExp(", code)
    return code.strip()


def train_full_model(df: pd.DataFrame, feature_cols: list):
    """Train final regularized binary LightGBM on 100% of S2_MACD setups."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=6,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=42,
        verbose=-1
    )
    model.fit(df[feature_cols], df["target_win"])
    return model


def main():
    m1, m5 = load_data(M1_PATH, M5_PATH)

    print("Preparing 100% S2_MACD training data for BUY and SELL models...")
    buy_df = simulate_m5_setups_on_m1(m5, m1, "BUY")
    sell_df = simulate_m5_setups_on_m1(m5, m1, "SELL")

    del m1, m5
    gc.collect()

    feature_cols = [
        "m5_body_ratio", "m5_upper_wick", "m5_lower_wick",
        "m5_range_atr", "m5_vol_ratio", "m5_trend_gap",
        "m5_macd_norm", "m5_hist_norm", "m5_adx14", "m5_rsi14",
        "hour", "dayofweek"
    ]

    print(f"Training Final S2_MACD BUY Model on {len(buy_df):,} setups...")
    buy_model = train_full_model(buy_df, feature_cols)

    print(f"Training Final S2_MACD SELL Model on {len(sell_df):,} setups...")
    sell_model = train_full_model(sell_df, feature_cols)

    print("Exporting models to MQL4 via m2cgen...")
    buy_mql4 = convert_c_to_mql4(m2c.export_to_c(buy_model), "AI_MACD_BuyProb")
    sell_mql4 = convert_c_to_mql4(m2c.export_to_c(sell_model), "AI_MACD_SellProb")

    header_comment = """//+------------------------------------------------------------------+
//|                                    XAUUSD_M5_MACD_AI_Models.mqh  |
//|                    Auto-generated S2_MACD LightGBM via m2cgen    |
//+------------------------------------------------------------------+
// Feature array order (double input[12]):
//   input[0]  = m5_body_ratio   : (Close - Open) / (High - Low)
//   input[1]  = m5_upper_wick   : (High - MathMax(Open, Close)) / (High - Low)
//   input[2]  = m5_lower_wick   : (MathMin(Open, Close) - Low) / (High - Low)
//   input[3]  = m5_range_atr    : (High - Low) / M5_ATR(14)
//   input[4]  = m5_vol_ratio    : M5_ATR(14) / M5_ATR(50)
//   input[5]  = m5_trend_gap    : (M5_EMA20 - M5_EMA50) / M5_ATR(14)
//   input[6]  = m5_macd_norm    : M5_MACD_Main / M5_ATR(14)
//   input[7]  = m5_hist_norm    : (M5_MACD_Main - M5_MACD_Sig) / M5_ATR(14)
//   input[8]  = m5_adx14        : M5_ADX(14)
//   input[9]  = m5_rsi14        : M5_RSI(14)
//   input[10] = hour            : Hour of bar close (0-23)
//   input[11] = dayofweek       : Day of week (0=Mon .. 4=Fri, Python format)
//
// Recommended threshold from 5-Fold WFV: Prob >= 0.45 (PF 1.38, MDD 15.5R)
//+------------------------------------------------------------------+
#property strict

"""

    with open(OUTPUT_MQH, "w", encoding="utf-8") as f:
        f.write(header_comment)
        f.write("// === S2_MACD BUY MODEL ===\n")
        f.write(buy_mql4 + "\n\n")
        f.write("// === S2_MACD SELL MODEL ===\n")
        f.write(sell_mql4 + "\n")

    print(f"\nSUCCESS! Exported MQL4 header file to: {OUTPUT_MQH}")
    print(f"Sample BUY Prob (last 3 rows):  {np.round(buy_model.predict_proba(buy_df[feature_cols].tail(3))[:, 1], 4)}")
    print(f"Sample SELL Prob (last 3 rows): {np.round(sell_model.predict_proba(sell_df[feature_cols].tail(3))[:, 1], 4)}")


if __name__ == "__main__":
    main()