"""Load XAUUSD_M5.csv in the common MT4 / MT5 export formats and build higher timeframes."""
import numpy as np
import pandas as pd

_ALIASES = {
    "date": "date", "time": "time",
    "datetime": "datetime", "timestamp": "datetime", "gmt time": "datetime", "local time": "datetime",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "tickvol": "volume", "tick_volume": "volume", "volume": "volume", "vol": "volume",
    "spread": "spread",
}


def _sniff_sep(line: str) -> str:
    if "\t" in line:
        return "\t"
    if line.count(";") > line.count(","):
        return ";"
    return ","


def load_m5_csv(path: str) -> pd.DataFrame:
    """Return a DataFrame indexed by bar-open time with open/high/low/close/volume[/spread].

    Supported:
      * MT4 History Center export (no header): 2024.01.02,00:05,2063.1,2064.0,2062.8,2063.5,412
      * MT5 export (tab, header):  <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
      * Generic header CSV with Date/Time or a single datetime column.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip()
    sep = _sniff_sep(first)
    has_header = not first[:1].isdigit()

    if has_header:
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
        df.columns = [_ALIASES.get(c.strip().strip("<>").lower(), c.strip().strip("<>").lower())
                      for c in df.columns]
    else:
        df = pd.read_csv(path, sep=sep, header=None)
        n = df.shape[1]
        if n >= 7:
            df = df.iloc[:, :7]
            df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
        elif n == 6:
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        else:
            raise ValueError(f"Unrecognised CSV layout with {n} columns: {first!r}")

    if "datetime" not in df.columns:
        if "date" not in df.columns or "time" not in df.columns:
            raise ValueError(f"CSV needs Date+Time or a datetime column, got {list(df.columns)}")
        dt = df["date"].astype(str).str.replace(".", "-", regex=False) + " " + df["time"].astype(str)
    else:
        dt = df["datetime"].astype(str).str.replace(".", "-", regex=False)
    df.index = pd.to_datetime(dt, format="mixed")
    df.index.name = "time"

    keep = ["open", "high", "low", "close"] + [c for c in ("volume", "spread") if c in df.columns]
    df = df[keep].astype(float)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])

    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        print(f"[data] warning: {int(bad.sum())} bars with inconsistent OHLC - dropped")
        df = df[~bad]
    return df


def guess_point(df: pd.DataFrame) -> float:
    """Guess MODE_POINT from the number of decimals in the prices."""
    s = df["close"].head(5000).map(lambda x: f"{x:.5f}".rstrip("0"))
    dec = s.str.split(".").str[1].str.len().fillna(0).max()
    return 10.0 ** -max(int(dec), 2)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df[list(agg)].resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])
