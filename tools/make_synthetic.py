"""Generate a synthetic XAUUSD_M5.csv (MT4 export format) for smoke-testing the pipeline.
NOT market data - results on it mean nothing."""
import sys
import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "data/XAUUSD_M5_synthetic.csv"
days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
rng = np.random.default_rng(7)
idx = pd.date_range("2025-01-01", periods=days * 288, freq="5min")
idx = idx[idx.dayofweek < 5]
n = len(idx)
hour = idx.hour.to_numpy()
vol = 0.6 + 0.8 * ((hour >= 7) & (hour <= 18))          # intraday volatility pattern
regime = np.repeat(rng.normal(0, 0.04, n // 500 + 1), 500)[:n]  # trending drift regimes
ret = regime + rng.standard_t(4, n) * vol * 0.9
close = 2000 + np.cumsum(ret)
open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.05, n)
hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.6, n)) * vol
lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.6, n)) * vol
v = (rng.poisson(300, n) * vol).astype(int)
df = pd.DataFrame({"d": idx.strftime("%Y.%m.%d"), "t": idx.strftime("%H:%M"),
                   "o": open_.round(3), "h": hi.round(3), "l": lo.round(3), "c": close.round(3), "v": v})
df.to_csv(out, header=False, index=False)
print(f"wrote {n} bars -> {out}")
