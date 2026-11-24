"""Features known at decision time (close of bar t) - no look-ahead."""
import numpy as np
import pandas as pd

from .data import resample
from .indicators import atr, ema, macd_main, rsi
from .sss import Market


def _htf_asof(df_htf: pd.DataFrame, cols, dec_time: pd.DatetimeIndex, tf: str) -> pd.DataFrame:
    """Value of the last *completed* higher-TF bar at each decision time."""
    closed_at = df_htf.index + pd.Timedelta(tf)
    src = df_htf[cols].copy()
    src.index = closed_at
    pos = np.searchsorted(src.index.values, dec_time.values, side="right") - 1
    out = src.iloc[np.clip(pos, 0, None)].to_numpy(dtype=float, copy=True)
    out[pos < 0] = np.nan
    return pd.DataFrame(out, columns=cols, index=dec_time)


def bar_features(m: Market) -> pd.DataFrame:
    df = m.df
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    a = pd.Series(m.atr, index=df.index)
    f = pd.DataFrame(index=df.index)
    f["atr"] = a
    f["atr_rank_288"] = a.rolling(288, min_periods=50).rank(pct=True)
    f["ema_slope_atr"] = (pd.Series(m.ema, index=df.index).diff(2)) / a
    f["dist_ema_atr"] = (c - m.ema) / a
    f["ema50_dist_atr"] = (c - ema(c, 50)) / a
    f["ema200_dist_atr"] = (c - ema(c, 200)) / a
    f["rsi14"] = rsi(c, 14)
    f["macd_atr"] = macd_main(c) / a
    for k in (1, 3, 6, 12, 24):
        f[f"ret_{k}_atr"] = c.diff(k) / a
    f["range12_atr"] = (h.rolling(12).max() - l.rolling(12).min()) / a
    f["body_atr"] = (c - o).abs() / a
    f["upper_wick_atr"] = (h - np.maximum(o, c)) / a
    f["lower_wick_atr"] = (np.minimum(o, c) - l) / a
    vol = df["volume"]
    f["vol_ratio20"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    f["spread_atr"] = pd.Series(m.spread, index=df.index) / a

    # M15 / H1 / H4 context (completed bars only)
    for tf, rule in (("m15", "15min"), ("h1", "1h"), ("h4", "4h")):
        x = resample(df, rule)
        xa = atr(x, 14)
        x = x.assign(
            **{f"{tf}_ema_slope": ema(x["close"], 50).diff(5) / xa,
               f"{tf}_dist_ema50": (x["close"] - ema(x["close"], 50)) / xa,
               f"{tf}_rsi": rsi(x["close"], 14),
               f"{tf}_macd": macd_main(x["close"]) / xa})
        cols = [f"{tf}_ema_slope", f"{tf}_dist_ema50", f"{tf}_rsi", f"{tf}_macd"]
        hf = _htf_asof(x, cols, m.dec_time, rule)
        hf.index = df.index
        f = f.join(hf)

    f["box_active"] = m.box_ok.astype(float)
    bh = m.box_hi - m.box_lo
    f["box_height_atr"] = np.where(m.box_ok, bh / m.atr, np.nan)
    f["box_pos"] = np.where(m.box_ok & (bh > 0), (c.to_numpy() - m.box_lo) / np.where(bh > 0, bh, 1), np.nan)
    f["vol_guard_ok"] = m.vol_ok.astype(float)
    f["hour"] = m.dec_time.hour
    f["dow"] = m.dec_time.dayofweek
    return f


DIR_SIGNED = ["ema_slope_atr", "dist_ema_atr", "ema50_dist_atr", "ema200_dist_atr", "macd_atr",
              "ret_1_atr", "ret_3_atr", "ret_6_atr", "ret_12_atr", "ret_24_atr",
              "m15_ema_slope", "m15_dist_ema50", "m15_macd", "h1_ema_slope", "h1_dist_ema50", "h1_macd",
              "h4_ema_slope", "h4_dist_ema50", "h4_macd"]


def candidate_features(m: Market, bf: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-candidate features. Trend-type features are multiplied by direction so that
    "with the trade" is always positive (lets one model serve BUY and SELL)."""
    X = bf.iloc[rows["t"].to_numpy()].reset_index(drop=True).copy()
    d = rows["dir"].to_numpy()
    for col in DIR_SIGNED:
        X[col] = X[col] * d
    X["rsi14"] = np.where(d == 1, X["rsi14"], 100 - X["rsi14"])
    for tf in ("m15", "h1", "h4"):
        X[f"{tf}_rsi"] = np.where(d == 1, X[f"{tf}_rsi"], 100 - X[f"{tf}_rsi"])
    X["box_pos"] = np.where(d == 1, X["box_pos"], 1 - X["box_pos"])
    upw, low = X.pop("upper_wick_atr"), X.pop("lower_wick_atr")
    X["wick_against_atr"] = np.where(d == 1, upw, low)
    X["wick_with_atr"] = np.where(d == 1, low, upw)

    t, k, pb = rows["t"].to_numpy(), rows["k"].to_numpy(), rows["pb_bar"].to_numpy()
    a = m.atr[t]
    X["dir"] = d
    X["bars_since_breakout"] = t - k
    X["bars_since_pullback"] = t - pb
    X["breakout_body_atr"] = np.abs(m.c[k] - m.o[k]) / a
    X["breakout_excess_atr"] = np.where(d == 1, m.c[k] - m.sw_hi[k], m.sw_lo[k] - m.c[k]) / a
    X["pullback_depth_atr"] = np.abs(m.c[t] - rows["pb_extreme"].to_numpy()) / a
    X["sl_dist_atr"] = rows["sl_dist"].to_numpy() / a
    X["sl_raw_dist_atr"] = rows["sl_raw_dist"].to_numpy() / a
    X["sl_clamped"] = (np.abs(rows["sl_dist"].to_numpy() - rows["sl_raw_dist"].to_numpy()) > 1e-9).astype(float)
    X["session"] = rows["session"].to_numpy()
    X["is_rebound"] = (rows["kind"] == "reject_rebound").astype(float).to_numpy()
    return X
