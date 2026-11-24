"""MT4-compatible indicators (iMA EMA, iATR, iMACD main line)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    # MT4 EMA: alpha = 2/(n+1), seeded with the first value (== adjust=False)
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # MT4 iATR = simple moving average of True Range
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr.rolling(n).mean()


def macd_main(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(close, fast) - ema(close, slow)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def filtered_avg_body(o: np.ndarray, c: np.ndarray, period: int = 20) -> np.ndarray:
    """GetAvgBodySize(): mean body of the last `period` closed bars, excluding >2x / <0.1x raw-avg outliers.

    Value at index t uses bars t-period+1..t (i.e. bar t is MT4's shift 1).
    """
    body = np.abs(c - o)
    n = len(body)
    out = np.full(n, np.nan)
    if n < period:
        return out
    win = np.lib.stride_tricks.sliding_window_view(body, period)
    raw = win.mean(axis=1)
    mask = (win <= raw[:, None] * 2.0) & (win >= raw[:, None] * 0.1)
    cnt = mask.sum(axis=1)
    filt = np.where(cnt > 0, (win * mask).sum(axis=1) / np.maximum(cnt, 1), raw)
    out[period - 1:] = filt
    return out
