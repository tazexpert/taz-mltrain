"""Compare EA v7.22 SSS back-test against proposed fixes on the same CSV.

    python tools/whatif.py taz_ml/XAUUSD_M5.csv
"""
import sys
import pandas as pd
sys.path.insert(0, ".")
from taz_ml.config import EAConfig
from taz_ml.data import load_m5_csv, guess_point
from taz_ml.sss import prepare, scan_pullback_candidates, run_ea
from taz_ml.train import summarize

df = load_m5_csv(sys.argv[1])
VARIANTS = {
    "EA v7.22 (as is)": {},
    "no Reject+Rebound orders (S/R filter kept)": dict(rebound_trades_disabled=True),
    "RR off entirely (also disables S/R filter, as in EA)": dict(rebound_enabled=False),
    "min SL $7 enforced": dict(sl_min_enforced=True),
    "no RR orders + min SL enforced": dict(rebound_trades_disabled=True, sl_min_enforced=True),
}
rows = []
for name, kw in VARIANTS.items():
    cfg = EAConfig(point=guess_point(df), **kw)
    m = prepare(df, cfg)
    trades, _ = run_ea(m, cfg, scan_pullback_candidates(m, cfg))
    t = pd.DataFrame(trades)
    s = summarize(t["set_r"])
    half = len(t) // 2
    s["1st_half_avg_r"] = round(t["set_r"].iloc[:half].mean(), 4)
    s["2nd_half_avg_r"] = round(t["set_r"].iloc[half:].mean(), 4)
    s["avg_sl_$"] = round(t["sl_dist"].mean(), 2)
    rows.append(dict(variant=name, **s))
print(pd.DataFrame(rows).to_string(index=False))
