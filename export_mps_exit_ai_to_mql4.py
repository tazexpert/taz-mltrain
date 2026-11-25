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
# Configuration Parameters (MPS v10.13)
# ==========================================
M1_PATH = "XAUUSD_M1.csv"
M5_PATH = "XAUUSD_M5.csv"
OUTPUT_MQH = "XAUUSD_M5_MPS_Exit_AI.mqh"

SPREAD = 0.25
BE_BUFFER = 0.5          # MPS_BEBufferPts
RR1 = 1.0                # InpRiskRewardRatio
RR2 = 2.5                # InpRiskRewardRatio2
MAX_HOLD_MIN = 288 * 5   # MPS_MaxHoldBars = 288 M5 bars (24h)
FRI_NO_ENTRY_MIN = 1200  # 20:00 Friday
FRI_CLOSE_MIN = 1305     # 21:45 Friday


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
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = wilder_atr(df, period)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def load_data():
    """Load M1 and M5 data and compute exact MPS & Exit AI features."""
    print("Loading M1 and M5 CSV files...")
    m1 = pd.read_csv(
        M1_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    m5 = pd.read_csv(
        M5_PATH, usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]
    ).sort_values("time").reset_index(drop=True)

    # H1 RSI(14) for MPS trend filter
    h1 = m5.set_index("time").resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    h1["h1_rsi14"] = wilder_rsi(h1["close"], 14)
    h1["h1_ready_time"] = h1["time"] + pd.Timedelta(hours=1)
    h1_r = h1[["h1_ready_time", "h1_rsi14"]]

    # M5 Indicators for MPS Entry
    m5["ema21"] = m5["close"].ewm(span=21, adjust=False).mean()
    m5["ema50"] = m5["close"].ewm(span=50, adjust=False).mean()
    m5["ema200"] = m5["close"].ewm(span=200, adjust=False).mean()
    m5["w_atr14"] = wilder_atr(m5, 14)
    m5["w_rsi14"] = wilder_rsi(m5["close"], 14)

    # M5 Indicators for Exit AI (reusing AIP_SimpleATR, AIP_SimpleRSI, iMACD, iADX)
    m5["adx14"] = compute_adx(m5, 14)
    ema12 = m5["close"].ewm(span=12, adjust=False).mean()
    ema26 = m5["close"].ewm(span=26, adjust=False).mean()
    m5["macd"] = ema12 - ema26
    m5["macd_sig"] = m5["macd"].rolling(9).mean()  # MT4 iMACD signal uses SMA(9)
    m5["macd_hist"] = m5["macd"] - m5["macd_sig"]

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
    m5["dayofweek"] = m5["entry_time"].dt.day_of_week
    m5["min_of_day"] = m5["hour"] * 60 + m5["minute"]

    m5 = pd.merge_asof(m5, h1_r, left_on="entry_time", right_on="h1_ready_time", direction="backward")
    m5 = m5[(m5["entry_time"] >= m1["time"].min()) & (m5["entry_time"] <= m1["time"].max())].dropna().reset_index(drop=True)

    del h1, h1_r
    gc.collect()
    return m1, m5


def extract_mps_snapshots(m1: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """Generate all MPS signals and extract closed-M5 in-trade snapshots for training."""
    fri_no_entry = (m5["dayofweek"] == 4) & (m5["min_of_day"] >= FRI_NO_ENTRY_MIN)
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

    sig_indices = np.where(mps_buy | mps_sell)[0]
    print(f"Found {len(sig_indices):,} MPS setups. Extracting M5 in-trade snapshots...")

    m1_times = m1["time"].values
    m1_opens = m1["open"].values
    m1_highs = m1["high"].values
    m1_lows = m1["low"].values
    m1_closes = m1["close"].values
    m1_dow = m1["time"].dt.day_of_week.values
    m1_mod = (m1["time"].dt.hour * 60 + m1["time"].dt.minute).values
    n_m1 = len(m1_times)

    m5_times = m5["entry_time"].values
    m5_adx = m5["adx14"].values
    m5_rsi = m5["s_rsi14"].values
    m5_hist = m5["hist_norm"].values
    m5_body = m5["body_ratio"].values

    snapshots = []

    for idx in sig_indices:
        entry_time = m5["entry_time"].iloc[idx].to_datetime64()
        direction = 1 if mps_buy.iloc[idx] else -1
        sl_dist = float(np.clip(m5["w_atr14"].iloc[idx] * 1.0, 6.5, 9.0))

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
        mfe_r, mae_r = 0.0, 0.0
        final_r = np.nan
        trade_snaps = []

        for j in range(s_idx, n_m1):
            t_now = m1_times[j]
            if direction == 1:
                fav_r = (m1_highs[j] - entry) / sl_dist
                adv_r = (m1_lows[j] - entry) / sl_dist
                cur_px = m1_closes[j]
                if not hit_tp1:
                    if m1_lows[j] <= sl_p:
                        final_r = -2.0
                        break
                    if m1_highs[j] >= tp1_p:
                        hit_tp1 = True
                        o2_sl = entry + BE_BUFFER
                        if m1_highs[j] >= tp2_p:
                            final_r = RR1 + RR2
                            break
                else:
                    if m1_lows[j] <= o2_sl:
                        final_r = RR1 + (o2_sl - entry) / sl_dist
                        break
                    if m1_highs[j] >= tp2_p:
                        final_r = RR1 + RR2
                        break
            else:
                ah, al = m1_highs[j] + SPREAD, m1_lows[j] + SPREAD
                fav_r = (entry - al) / sl_dist
                adv_r = (entry - ah) / sl_dist
                cur_px = m1_closes[j] + SPREAD
                if not hit_tp1:
                    if ah >= sl_p:
                        final_r = -2.0
                        break
                    if al <= tp1_p:
                        hit_tp1 = True
                        o2_sl = entry - BE_BUFFER
                        if al <= tp2_p:
                            final_r = RR1 + RR2
                            break
                else:
                    if ah >= o2_sl:
                        final_r = RR1 + (entry - o2_sl) / sl_dist
                        break
                    if al <= tp2_p:
                        final_r = RR1 + RR2
                        break

            mfe_r = max(mfe_r, fav_r)
            mae_r = min(mae_r, adv_r)

            if (m1_dow[j] == 4 and m1_mod[j] >= FRI_CLOSE_MIN) or (t_now >= max_exit_time):
                leg_r = direction * (cur_px - entry) / sl_dist
                final_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                break

            if m1_mod[j] % 5 == 4:
                next_m5_t = t_now + np.timedelta64(1, "m")
                m5_idx = np.searchsorted(m5_times, next_m5_t, side="right") - 1
                if m5_idx >= 0:
                    bars_held = int((next_m5_t - entry_time) / np.timedelta64(5, "m"))
                    leg_r = direction * (cur_px - entry) / sl_dist
                    curr_total_r = (RR1 + leg_r) if hit_tp1 else (2.0 * leg_r)
                    tp_rem_r = (RR2 - leg_r) if hit_tp1 else (RR1 - leg_r)
                    drawback_r = mfe_r - leg_r

                    rsi_dir = m5_rsi[m5_idx] if direction == 1 else (100.0 - m5_rsi[m5_idx])
                    hist_dir = direction * m5_hist[m5_idx]
                    body_dir = direction * m5_body[m5_idx]

                    trade_snaps.append({
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
            for s in trade_snaps:
                s["target_hold"] = 1 if final_r > s["curr_total_r"] else 0
                snapshots.append(s)

    return pd.DataFrame(snapshots)


def convert_c_to_mql4_clean(c_code: str, score_func: str, sigmoid_func: str) -> str:
    """
    Convert m2cgen C output into clean, compilable MQL4 code.
    Fixes all 3 known m2cgen MQL4 issues:
      1) Renames sigmoid() and score() to avoid name collisions with existing models.
      2) Converts C pointer parameters to MQL4 array references.
      3) Replaces C99 compound-literal memcpy() with direct output[0]/output[1] assignments.
    """
    code = re.sub(r"#include\s*<[^>]+>\s*", "", c_code)
    code = re.sub(r"\bsigmoid\s*\(", f"{sigmoid_func}(", code)
    code = code.replace("double * input, double * output", "const double &input[], double &output[]")
    code = code.replace("double * input", "const double &input[]")
    code = re.sub(r"\bvoid\s+score\s*\(", f"void {score_func}(", code)
    code = re.sub(r"\bexp\s*\(", "MathExp(", code)

    # Replace: memcpy(output, (double[]){1.0 - varX, varX}, 2 * sizeof(double));
    def replace_memcpy(match):
        expr0 = match.group(1).strip()
        expr1 = match.group(2).strip()
        return f"output[0] = {expr0};\n    output[1] = {expr1};"

    code = re.sub(
        r"memcpy\s*\(\s*output\s*,\s*\(double\[\]\)\s*\{\s*([^,]+)\s*,\s*([^}]+)\s*\}\s*,\s*2\s*\*\s*sizeof\s*\(\s*double\s*\)\s*\)\s*;",
        replace_memcpy,
        code
    )
    return code.strip()


def main():
    m1, m5 = load_data()
    snaps_df = extract_mps_snapshots(m1, m5)
    print(f"Total MPS training snapshots: {len(snaps_df):,} (Hold=1 rate: {snaps_df['target_hold'].mean()*100:.2f}%)")

    del m1, m5
    gc.collect()

    feat_cols = [
        "bars_held", "hit_tp1", "curr_r", "tp_rem_r", "mfe_r", "mae_r", "drawback_r",
        "adx14", "rsi_dir", "hist_dir", "body_dir"
    ]

    print("Training Final MPS Exit Manager AI on 100% of MPS snapshots...")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        n_jobs=2,
        random_state=42,
        verbose=-1
    )
    model.fit(snaps_df[feat_cols].values, snaps_df["target_hold"].values)

    print("Exporting to MQL4 via m2cgen...")
    raw_c = m2c.export_to_c(model)
    mql4_code = convert_c_to_mql4_clean(raw_c, "AI_MPS_ExitScore", "MPS_ExitSigmoid")

    header_comment = """//+------------------------------------------------------------------+
//|                                     XAUUSD_M5_MPS_Exit_AI.mqh    |
//|          Auto-generated MPS-Only Exit Manager AI via m2cgen      |
//|          Pre-fixed for MQL4 (no name collision, no memcpy)       |
//+------------------------------------------------------------------+
// Feature array order (double input[11]) - evaluated at each closed M5 bar:
//   input[0]  = bars_held   : M5 bars elapsed since OrderOpenTime()
//   input[1]  = hit_tp1     : 1.0 if TP1 leg already closed (runner only), else 0.0
//   input[2]  = curr_r      : dir * (closePrice - openPrice) / origSLDist
//   input[3]  = tp_rem_r    : (RR2 - curr_r) if hit_tp1 else (RR1 - curr_r)
//   input[4]  = mfe_r       : Max Favorable Excursion in R since open (>= 0)
//   input[5]  = mae_r       : Max Adverse Excursion in R since open (<= 0)
//   input[6]  = drawback_r  : mfe_r - curr_r
//   input[7]  = adx14       : M5 iADX(14, shift=1)
//   input[8]  = rsi_dir     : AIP_SimpleRSI(14,1) if BUY else (100 - AIP_SimpleRSI(14,1))
//   input[9]  = hist_dir    : dir * (iMACD_Main - iMACD_Sig) / AIP_SimpleATR(14,1)
//   input[10] = body_dir    : dir * (Close[1] - Open[1]) / (High[1] - Low[1]) on M5
//
// Recommended Hybrid Policy (MPS_Order ONLY - do NOT apply to AIP or HTF):
//   - P(Hold) < 0.25 : Close open MPS leg(s) at market immediately
//   - P(Hold) < 0.40 : Tighten SL to BE+Buffer (if in profit) or -0.5R (if in loss)
//+------------------------------------------------------------------+
#property strict

"""

    wrapper_func = """
// Wrapper returning P(Hold beats closing now) in [0.0, 1.0]
double AI_MPS_HoldProb(const double &input[])
{
   double out[2];
   AI_MPS_ExitScore(input, out);
   return out[1];
}
"""

    with open(OUTPUT_MQH, "w", encoding="utf-8") as f:
        f.write(header_comment)
        f.write(mql4_code + "\n")
        f.write(wrapper_func)

    sample_probs = np.round(model.predict_proba(snaps_df[feat_cols].tail(5).values)[:, 1], 4)
    print(f"\nSUCCESS! Exported clean MQL4 Exit AI file to: {OUTPUT_MQH}")
    print(f"Sample P(Hold) (last 5 snapshots): {sample_probs}")


if __name__ == "__main__":
    main()