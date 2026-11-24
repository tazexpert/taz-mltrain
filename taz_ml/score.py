"""Score the most recent SSS entry candidates in a CSV with a trained model.

    python -m taz_ml.score --csv data/XAUUSD_M5.csv --model output/model.joblib --last 20
"""
import argparse

import joblib
import pandas as pd

from .config import EAConfig
from .data import load_m5_csv
from .train import build_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="output/model.joblib")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    bundle = joblib.load(args.model)
    cfg = EAConfig(**{k: (tuple(map(tuple, v)) if k == "sessions" else v) for k, v in bundle["config"].items()})
    df = load_m5_csv(args.csv)
    _, ds, X, _ = build_dataset(df, cfg, verbose=False)
    ds["p_win"] = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    ds["take"] = ds["p_win"] >= args.threshold
    cols = ["time", "dir", "kind", "session", "ea_status", "entry", "sl", "sl_dist", "p_win", "take", "set_r"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ds[cols].tail(args.last).to_string(index=False))
    print("\nset_r of the newest rows may be incomplete (trade not finished in the data).")


if __name__ == "__main__":
    main()
