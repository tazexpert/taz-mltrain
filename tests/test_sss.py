"""Sanity tests: the vectorised SSS port must match a literal line-by-line port of the MQ4."""
import os
import numpy as np
import pandas as pd
import pytest

from taz_ml.config import EAConfig
from taz_ml import sss

HERE = os.path.dirname(__file__)


def _synthetic(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-03-03", periods=n, freq="5min")
    ret = np.repeat(rng.normal(0, 0.05, n // 300 + 1), 300)[:n] + rng.standard_t(4, n) * 0.8
    c = 2000 + np.cumsum(ret)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 100.0}, index=idx)


def _literal_signal_at(m, cfg, k):
    """SSS_GetSignalAt(shift) with explicit loops (bar k == shift)."""
    pt = cfg.point * 10
    a = m.atr[k]
    if not (cfg.atr_min_pts * pt <= a <= cfg.atr_max_pts * pt):
        return 0, None, None
    slope = m.ema[k] - m.ema[k - 2]
    swH = m.l[k - 1]
    swL = m.h[k - 1]
    for i in range(k - 1, k - cfg.swing_bars - 1, -1):
        swH = max(swH, m.h[i])
        swL = min(swL, m.l[i])
    b1, b2 = abs(m.c[k] - m.o[k]), abs(m.c[k - 1] - m.o[k - 1])
    eng = b1 >= b2 * cfg.engulf_ratio
    if slope > 0 and m.c[k] > swH and m.c[k] > m.o[k] and eng:
        return 1, swH, swL
    if slope < 0 and m.c[k] < swL and m.c[k] < m.o[k] and eng:
        return -1, swH, swL
    return 0, swH, swL


def test_signal_at_matches_literal():
    cfg = EAConfig()
    m = sss.prepare(_synthetic(), cfg)
    for k in range(40, len(m.c)):
        s, swH, swL = _literal_signal_at(m, cfg, k)
        assert s == m.sig[k], k
        if s:
            assert swH == pytest.approx(m.sw_hi[k]) and swL == pytest.approx(m.sw_lo[k])


def test_pullback_candidates_match_literal_loop():
    """Literal SSS_CheckPullbackEntry() called on every bar with g_SSS_LastPullbackTradedTime."""
    cfg = EAConfig()
    m = sss.prepare(_synthetic(), cfg)
    max_lb = cfg.pullback_max_wait + cfg.swing_bars + 5
    last_used = -1
    lit = []
    for t in range(40, len(m.c) - 1):
        for ss in range(3, max_lb + 1):
            k = t - ss + 1
            if m.sig[k] == 0:
                continue
            if k <= last_used:
                break
            r = sss.pullback_scan(m, k, t, cfg.pullback_max_wait)
            if r:
                last_used = k
                lit.append((t, k, r[0]))
            break
    fast = [(c["t"], c["k"], c["dir"]) for c in sss.scan_pullback_candidates(m, cfg)]
    assert fast == lit
    assert len(fast) > 20


def test_no_lookahead_in_box_and_features():
    """Changing the future must not change anything computed for earlier decision bars."""
    from taz_ml.features import bar_features
    cfg = EAConfig()
    df = _synthetic(8000)
    m1 = sss.prepare(df, cfg)
    df2 = df.copy()
    df2.iloc[6000:, :4] += 50.0  # shock the future
    df2["high"] = df2[["open", "high", "close"]].max(axis=1)
    df2["low"] = df2[["open", "low", "close"]].min(axis=1)
    m2 = sss.prepare(df2, cfg)
    cut = 5990
    assert np.array_equal(m1.box_ok[:cut], m2.box_ok[:cut])
    assert np.array_equal(m1.sig[:cut], m2.sig[:cut])
    f1, f2 = bar_features(m1).iloc[:cut], bar_features(m2).iloc[:cut]
    pd.testing.assert_frame_equal(f1, f2)


def test_simulate_set_basic_tp_and_sl():
    cfg = EAConfig()
    cfg.news_enabled = False
    cfg.dr_enabled = False
    idx = pd.date_range("2025-03-03 03:00", periods=60, freq="5min")
    c = np.full(60, 2000.0)
    df = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": 1.0}, index=idx)
    df.iloc[45:, :] = 2000.0
    df.loc[idx[50], "high"] = 2020.0  # spike up -> TP1 and TP2 hit
    m = sss.prepare(df, cfg)
    out = sss.simulate_set(m, 40, 1, 2000.2, 1990.2, cfg)
    assert out["tp1_reason"] == "tp" and out["tp1_r"] == pytest.approx(1.0)
    assert out["tp2_reason"] == "tp" and out["tp2_r"] == pytest.approx(1.5)
    out = sss.simulate_set(m, 40, -1, 2000.0, 2010.0, cfg)
    assert out["tp1_reason"] == "sl" and out["tp1_r"] == pytest.approx(-1.0)  # Ask (Bid+spread) reaches SL -> filled at SL
