"""Build the SSS entry dataset from XAUUSD_M5.csv, back-test the EA logic and train a
model that predicts whether an SSS entry will win.

    python -m taz_ml.train --csv data/XAUUSD_M5.csv --out output
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from .config import EAConfig
from .data import load_m5_csv, guess_point
from .features import bar_features, candidate_features
from .sss import prepare, scan_pullback_candidates, run_ea, entry_sl, simulate_set, session_at

TAKEN = ("traded", "traded_rr")


def build_dataset(df: pd.DataFrame, cfg: EAConfig, verbose=True):
    t0 = time.time()
    m = prepare(df, cfg)
    cands = scan_pullback_candidates(m, cfg)
    trades, status = run_ea(m, cfg, cands)
    if verbose:
        print(f"[sss] bars={len(df):,} candidates={len(cands):,} ea_trades={len(trades):,} ({time.time()-t0:.1f}s)")

    # Label every candidate as if the order-set had been sent (TP1 + TP2)
    rows = []
    for c in cands:
        t, d = c["t"], c["dir"]
        entry, sl, raw, dist = entry_sl(m, d, c["pb_extreme"], t, cfg)
        sim = simulate_set(m, t, d, entry, sl, cfg, send_tp2=True)
        r = dict(c, kind="pullback", entry=entry, sl=sl, sl_raw_dist=raw, sl_dist=dist,
                 session=session_at(int(m.hhmm[t]), cfg), time=m.dec_time[t],
                 ea_status=status.get(t, "not_evaluated"))
        r.update(sim)
        rows.append(r)
    # Reject+Rebound trades that don't coincide with a pullback candidate bar
    cand_t = {c["t"] for c in cands}
    for tr in trades:
        if tr["kind"] == "reject_rebound" and tr["t"] not in cand_t:
            r = {k: v for k, v in tr.items() if k not in ("day", "tp2_sent", "same_bar_as_sr_block")}
            r["ea_status"] = "traded_rr"
            rows.append(r)
    ds = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    tr_df = pd.DataFrame(trades)

    bf = bar_features(m)
    X = candidate_features(m, bf, ds)
    ds["y"] = (ds["tp1_r"] > 0).astype(int)
    ds["exit_bar"] = ds["exit_key"] // 2
    return m, ds, X, tr_df


def summarize(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return dict(n=0)
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=int(len(r)), win_rate=round(float((r > 0).mean()), 4), avg_r=round(float(r.mean()), 4),
                total_r=round(float(r.sum()), 2), profit_factor=round(float(gp / gl), 3) if gl > 0 else None,
                max_dd_r=round(float(dd), 2))


def make_model(seed=42):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)


def walk_forward(ds: pd.DataFrame, X: pd.DataFrame, n_splits=5, min_train=200):
    """Expanding-window walk-forward with purging: a training row is used only if its
    trade had already closed before the first test bar (no label leakage)."""
    n = len(ds)
    oof = np.full(n, np.nan)
    folds = []
    edges = np.linspace(0, n, n_splits + 2).astype(int)[1:]
    for i in range(n_splits):
        te_lo, te_hi = edges[i], edges[i + 1]
        first_test_bar = ds["t"].iloc[te_lo]
        tr_mask = (np.arange(n) < te_lo) & (ds["exit_bar"].to_numpy() < first_test_bar)
        if tr_mask.sum() < min_train or ds.loc[tr_mask, "y"].nunique() < 2:
            continue
        mdl = make_model().fit(X[tr_mask], ds.loc[tr_mask, "y"])
        p = mdl.predict_proba(X.iloc[te_lo:te_hi])[:, 1]
        oof[te_lo:te_hi] = p
        yt = ds["y"].iloc[te_lo:te_hi]
        folds.append(dict(fold=i + 1, train=int(tr_mask.sum()), test=int(te_hi - te_lo),
                          test_from=str(ds["time"].iloc[te_lo]), test_to=str(ds["time"].iloc[te_hi - 1]),
                          auc=round(float(roc_auc_score(yt, p)), 4) if yt.nunique() > 1 else None,
                          base_rate=round(float(yt.mean()), 4)))
    return oof, folds, mdl if folds else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/XAUUSD_M5.csv")
    ap.add_argument("--out", default="output")
    ap.add_argument("--point", type=float, default=None, help="MODE_POINT (default: guess from CSV decimals)")
    ap.add_argument("--spread", type=float, default=None, help="Spread in $ if CSV has no spread column")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max-hold", type=int, default=None, help="Force-close after N M5 bars")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    df = load_m5_csv(args.csv)
    cfg = EAConfig()
    cfg.point = args.point or guess_point(df)
    if args.spread is not None:
        cfg.spread = args.spread
    if args.max_hold:
        cfg.max_hold_bars = args.max_hold
    print(f"[data] {len(df):,} bars {df.index[0]} -> {df.index[-1]}  point={cfg.point}  "
          f"spread={'csv' if 'spread' in df.columns else cfg.spread}")
    if cfg.point >= 0.01:
        print("[warn] 2-digit feed: EA point inputs (SL min 7000 pts, buffer 2000 pts) become $70/$20 - "
              "check that this matches the broker the EA runs on, or pass --point 0.001")

    m, ds, X, trades = build_dataset(df, cfg)
    if len(ds) < 100:
        raise SystemExit(f"Only {len(ds)} candidates - not enough to train. Need more history.")

    ds.to_csv(os.path.join(args.out, "dataset.csv"), index=False)
    pd.concat([ds[["time", "dir", "ea_status", "y", "set_r"]], X], axis=1).to_csv(
        os.path.join(args.out, "features.csv"), index=False)
    trades.to_csv(os.path.join(args.out, "ea_trades.csv"), index=False)

    report = {"data": dict(bars=len(df), start=str(df.index[0]), end=str(df.index[-1]), point=cfg.point),
              "config": cfg.to_dict()}
    report["candidates_status"] = ds["ea_status"].value_counts().to_dict()
    report["all_candidates"] = summarize(ds["set_r"])
    report["ea_backtest"] = summarize(trades["set_r"]) if len(trades) else {"n": 0}
    if len(trades):
        report["ea_backtest_by_kind"] = {k: summarize(g["set_r"]) for k, g in trades.groupby("kind")}
        report["ea_backtest_by_session"] = {int(k): summarize(g["set_r"]) for k, g in trades.groupby("session")}
        report["ea_exit_reasons_tp1"] = trades["tp1_reason"].value_counts().to_dict()
        if "same_bar_as_sr_block" in trades:
            report["rr_same_bar_as_sr_block"] = int(trades["same_bar_as_sr_block"].fillna(False).astype(bool).sum())
    report["sl_clamped_share"] = round(float((ds["sl_dist"] - ds["sl_raw_dist"]).abs().gt(1e-9).mean()), 4)
    report["sl_below_min_after_clamp"] = int((ds["sl_dist"] < cfg.p(cfg.sl_min_pts) - 1e-9).sum())

    # --- Model ---
    oof, folds, _ = walk_forward(ds, X, args.splits)
    report["walk_forward"] = folds
    mask = ~np.isnan(oof)
    if mask.sum() > 0 and ds.loc[mask, "y"].nunique() > 1:
        y, p = ds.loc[mask, "y"], oof[mask]
        report["oos_metrics"] = dict(n=int(mask.sum()), auc=round(float(roc_auc_score(y, p)), 4),
                                     logloss=round(float(log_loss(y, p)), 4),
                                     brier=round(float(brier_score_loss(y, p)), 4),
                                     base_rate=round(float(y.mean()), 4))
        ds["p_win"] = oof
        thr_tab = []
        taken = ds["ea_status"].isin(TAKEN)
        for thr in np.round(np.arange(0.30, 0.76, 0.05), 2):
            sel = mask & (ds["p_win"] >= thr)
            thr_tab.append(dict(threshold=float(thr), all=summarize(ds.loc[sel, "set_r"]),
                                ea_taken=summarize(ds.loc[sel & taken, "set_r"])))
        report["oos_threshold_table"] = thr_tab
        report["oos_baseline"] = dict(all=summarize(ds.loc[mask, "set_r"]),
                                      ea_taken=summarize(ds.loc[mask & taken, "set_r"]))
        ds[["time", "dir", "ea_status", "y", "set_r", "p_win"]].to_csv(
            os.path.join(args.out, "oos_predictions.csv"), index=False)

    final = make_model().fit(X, ds["y"])
    tail = slice(int(len(X) * 0.8), None)
    try:
        pi = permutation_importance(final, X.iloc[tail], ds["y"].iloc[tail], n_repeats=5,
                                    random_state=0, scoring="roc_auc")
        imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
        report["feature_importance_top15"] = imp.head(15).round(4).to_dict()
    except ValueError:
        pass
    joblib.dump(dict(model=final, features=list(X.columns), config=cfg.to_dict()),
                os.path.join(args.out, "model.joblib"))

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    _print_report(report)
    print(f"\n[out] saved dataset.csv, features.csv, ea_trades.csv, oos_predictions.csv, model.joblib, report.json -> {args.out}/")


def _print_report(r):
    print("\n=== EA back-test (SSS emulation, R units, 0.5*TP1 + 0.5*TP2) ===")
    print(json.dumps(r.get("ea_backtest"), indent=None))
    for k, v in r.get("ea_backtest_by_kind", {}).items():
        print(f"  {k:15s} {v}")
    print("candidate status:", r.get("candidates_status"))
    print("TP1 exit reasons:", r.get("ea_exit_reasons_tp1"))
    print(f"SL clamped share: {r.get('sl_clamped_share')}  SL < min after clamp: {r.get('sl_below_min_after_clamp')}")
    if "rr_same_bar_as_sr_block" in r:
        print("Reject+Rebound trades on the same bar the S/R filter blocked:", r["rr_same_bar_as_sr_block"])
    print("\n=== Walk-forward ===")
    for f in r.get("walk_forward", []):
        print("  ", f)
    if "oos_metrics" in r:
        print("OOS:", r["oos_metrics"])
        print("baseline:", r["oos_baseline"])
        print(f"{'thr':>5} | {'all: n  win  avgR  totR':>30} | {'EA-taken: n  win  avgR  totR':>34}")
        for row in r["oos_threshold_table"]:
            a, e = row["all"], row["ea_taken"]
            fa = f"{a.get('n',0):5d} {a.get('win_rate',0):.2f} {a.get('avg_r',0):+.3f} {a.get('total_r',0):+8.1f}"
            fe = f"{e.get('n',0):5d} {e.get('win_rate',0):.2f} {e.get('avg_r',0):+.3f} {e.get('total_r',0):+8.1f}"
            print(f"{row['threshold']:5.2f} | {fa:>30} | {fe:>34}")
    if "feature_importance_top15" in r:
        print("\nTop features (permutation AUC drop):")
        for k, v in r["feature_importance_top15"].items():
            print(f"  {k:24s} {v:+.4f}")


if __name__ == "__main__":
    main()
