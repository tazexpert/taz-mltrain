"""EA input parameters mirrored from `TAZ Expert Strategy-gemini.mq4` (v7.22).

Only the inputs that affect the SSS entry / exit logic are kept. Price distances in the
EA are given in *points*; `point` converts them to price. The EA comments assume a 3-digit
XAUUSD feed (2000 points = $2.00 -> point = 0.001). If your broker quotes 2 digits, the
EA's point-based inputs mean 10x larger dollar distances - set `point` accordingly.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class EAConfig:
    point: float = 0.001          # MODE_POINT of the feed (3-digit gold)
    spread: float = 0.20          # Spread in price ($) used when the CSV has no spread column

    # --- SSS core ---
    ema_slope: int = 21           # InpEMA_Slope
    swing_bars: int = 7           # InpSwingBars
    engulf_ratio: float = 1.0     # InpEngulfRatio
    atr_min_pts: int = 200        # InpATR_MinPts  (x point*10 in the EA!)
    atr_max_pts: int = 8000       # InpATR_MaxPts  (x point*10 in the EA!)
    pullback_max_wait: int = 18   # InpSSSPullbackMaxWaitBars
    rebound_enabled: bool = True  # InpEnableSSSRejectRebound
    rebound_max_wait: int = 12    # InpSSSReboundMaxWaitBars
    spike_body_ratio: float = 5.0  # InpSSSSpikeBodyRatio
    block_cooldown_bars: int = 6  # InpSSSBlockCooldownBars
    box_zone_pct: float = 0.0     # InpSSSBoxZonePct
    block_expire_pct: float = 300.0  # InpSSSBlockExpirePct

    # --- SL / TP ---
    sl_fixed_buffer_pts: int = 2000   # InpSSS_SL_FixedBuffer
    sl_min_pts: float = 7000          # InpM15S_ATR_MinPts
    sl_atr_fallback_mult: float = 2.0  # InpM15S_ATR_SL_Buf_Mult
    sl_max_pts: float = 15000         # InpM15S_ATR_MaxPts
    rr1: float = 1.0                  # InpRiskRewardRatio
    rr2: float = 1.5                  # InpRiskRewardRatio2

    # --- Auto SL ---
    autosl_enabled: bool = True
    autosl_trig1: float = 80.0
    autosl_lock1: float = 30.0
    autosl_trig2: float = 90.0
    autosl_lock2: float = 70.0
    tp2_trig1: float = 80.0
    tp2_lock1: float = 30.0
    tp2_trig2: float = 90.0
    tp2_lock2: float = 70.0
    dr_enabled: bool = True
    dr_band_pts: float = 1500
    dr_close_count: int = 3
    dr_lock_pct: float = 50.0

    # --- News protection (server time) ---
    news_enabled: bool = True
    news_hhmm: int = 1230
    news_mins_before: int = 5
    news_close_band_pct: float = 20.0
    news_lock_pct: float = 5.0

    # --- Sessions (server time HHMM, [start, end)) ---
    sessions: tuple = ((0, 225), (230, 455), (500, 725), (730, 955), (1000, 1200), (1500, 1630))
    max_orders: int = 2                  # InpMaxOrders
    max_sets_per_session: int = 2        # InpSSS_MaxOrdersPerSession
    reentry_min_time_left_pct: float = 40.0

    # --- M15 sideway box (IsMACDSidewaysAt) ---
    sw_lookback: int = 8
    ma_fast: int = 8
    ma_slow: int = 21
    ma_dist_atr_ratio: float = 4.0
    sw_atr_period: int = 14
    range_atr_ratio: float = 3.0
    macd_baseline_bars: int = 20
    macd_flat_ratio: float = 0.70
    body_atr_ratio: float = 0.5
    max_same_color_streak: int = 5
    macd_fast: int = 12
    macd_slow: int = 26

    # --- Guards ---
    vol_breaker: float = 2.0             # Guard 3 (H1 ATR)

    # --- Simulation ---
    max_hold_bars: int = 288 * 3         # Force-close after N M5 bars (3 days) if nothing hit

    def to_dict(self):
        return asdict(self)

    def p(self, pts: float) -> float:
        """Points -> price."""
        return pts * self.point
