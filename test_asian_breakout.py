import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "XAUUSD_M15.csv"
SPREAD = 0.20            # Gold spread in USD ($0.20)

ASIAN_START_HOUR = 0     # Asian session start (00:00 Server Time)
ASIAN_END_HOUR = 6       # Asian session end (includes 06:00 - 06:45 bars)
LONDON_START_HOUR = 7    # Breakout entry window start (07:00)
LONDON_END_HOUR = 11     # Breakout entry window end (11:45)
EOD_EXIT_HOUR = 21       # Force close open trade at 21:00 if neither TP/SL hit

# Dynamic Asian Range Filter based on Previous Day's Daily ATR(14)
MIN_ATR_MULT = 0.20      # Skip dead days (Asian range < 20% of Daily ATR)
MAX_ATR_MULT = 0.80      # Skip extreme days (Asian range > 80% of Daily ATR)

DAILY_EMA_PERIOD = 50    # Daily EMA period for macro trend filter
RR_RATIO = 1.5           # TP distance = 1.5x SL distance

# ==========================================
# 1. LOAD DATA & COMPUTE DAILY INDICATORS
# ==========================================
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load M15 OHLCV data and merge shifted Daily ATR(14) and Daily EMA trend."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['year'] = df['time'].dt.year
    df['hour'] = df['time'].dt.hour

    # Aggregate M15 bars into Daily OHLC
    daily = df.groupby('date').agg(
        d_open=('open', 'first'),
        d_high=('high', 'max'),
        d_low=('low', 'min'),
        d_close=('close', 'last')
    ).reset_index()

    # Calculate Daily True Range and ATR(14)
    d_hl = daily['d_high'] - daily['d_low']
    d_hc = np.abs(daily['d_high'] - daily['d_close'].shift(1))
    d_lc = np.abs(daily['d_low'] - daily['d_close'].shift(1))
    daily['d_tr'] = pd.concat([d_hl, d_hc, d_lc], axis=1).max(axis=1)
    daily['d_atr14'] = daily['d_tr'].rolling(window=14).mean()

    # Calculate Daily EMA for trend direction
    daily['d_ema'] = daily['d_close'].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()

    # Shift by 1 day to strictly avoid lookahead bias
    daily['prev_d_atr'] = daily['d_atr14'].shift(1)
    daily['prev_d_trend'] = np.where(
        daily['d_close'].shift(1) > daily['d_ema'].shift(1), 1, -1
    )
    # Invalidate trend before EMA/ATR warm-up completes
    daily.loc[daily['prev_d_atr'].isna(), 'prev_d_trend'] = 0

    # Merge previous day's metrics back to M15 dataframe
    df = df.merge(daily[['date', 'prev_d_atr', 'prev_d_trend']], on='date', how='left')
    return df

# ==========================================
# 2. BACKTEST ASIAN RANGE BREAKOUT
# ==========================================
def run_asian_breakout_backtest(df: pd.DataFrame, sl_range_mult: float, rr_ratio: float) -> pd.DataFrame:
    """Simulate 1 trade per day with Daily ATR range filter and Daily EMA trend filter."""
    trades = []

    for date, day_df in df.groupby('date'):
        prev_atr = day_df['prev_d_atr'].iloc[0]
        prev_trend = day_df['prev_d_trend'].iloc[0]

        if pd.isna(prev_atr) or prev_atr <= 0 or prev_trend == 0:
            continue

        # Extract Asian session bars
        asian_df = day_df[(day_df['hour'] >= ASIAN_START_HOUR) & (day_df['hour'] <= ASIAN_END_HOUR)]
        if len(asian_df) < 8:
            continue

        asian_high = asian_df['high'].max()
        asian_low = asian_df['low'].min()
        asian_range = asian_high - asian_low

        # Dynamic range filter relative to Daily ATR
        if asian_range < (prev_atr * MIN_ATR_MULT) or asian_range > (prev_atr * MAX_ATR_MULT):
            continue

        # Extract London breakout window bars
        london_df = day_df[(day_df['hour'] >= LONDON_START_HOUR) & (day_df['hour'] <= LONDON_END_HOUR)]
        if london_df.empty:
            continue

        # Find the first M15 bar closing outside the Asian range aligned with Daily EMA trend
        entry_idx = None
        direction = 0

        for idx, row in london_df.iterrows():
            if prev_trend == 1 and row['close'] > asian_high:
                entry_idx = idx
                direction = 1   # BUY (aligned with uptrend)
                break
            elif prev_trend == -1 and row['close'] < asian_low:
                entry_idx = idx
                direction = -1  # SELL (aligned with downtrend)
                break

        if entry_idx is None:
            continue

        # Define Entry, SL, and TP with real spread deduction
        sl_dist = asian_range * sl_range_mult
        tp_dist = sl_dist * rr_ratio

        entry_bar = day_df.loc[entry_idx]
        if direction == 1:  # BUY: Enter at Ask, exit at Bid
            entry_price = entry_bar['close'] + SPREAD
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:               # SELL: Enter at Bid, exit at Ask
            entry_price = entry_bar['close']
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Simulate trade path on subsequent bars until EOD_EXIT_HOUR
        future_bars = day_df[(day_df.index > entry_idx) & (day_df['hour'] <= EOD_EXIT_HOUR)]
        if future_bars.empty:
            continue

        r_result = None
        outcome = "EOD"

        for _, f_row in future_bars.iterrows():
            if direction == 1:
                if f_row['low'] <= sl_price:  # Conservative: check SL first
                    r_result = -1.0
                    outcome = "SL"
                    break
                if f_row['high'] >= tp_price:
                    r_result = rr_ratio
                    outcome = "TP"
                    break
            else:
                if (f_row['high'] + SPREAD) >= sl_price:  # Conservative: check SL first
                    r_result = -1.0
                    outcome = "SL"
                    break
                if (f_row['low'] + SPREAD) <= tp_price:
                    r_result = rr_ratio
                    outcome = "TP"
                    break

        # Close at EOD if neither TP nor SL was hit
        if r_result is None:
            last_close = future_bars.iloc[-1]['close']
            if direction == 1:
                pnl = last_close - entry_price
            else:
                pnl = entry_price - (last_close + SPREAD)
            r_result = pnl / sl_dist

        trades.append({
            'date': date,
            'year': entry_bar['year'],
            'direction': 'BUY' if direction == 1 else 'SELL',
            'asian_range': asian_range,
            'range_atr_ratio': asian_range / prev_atr,
            'outcome': outcome,
            'r_result': r_result
        })

    return pd.DataFrame(trades)

# ==========================================
# 3. PERFORMANCE METRICS & COMPARISON REPORT
# ==========================================
def calc_max_dd(r_series: pd.Series) -> float:
    """Calculate maximum drawdown in R-multiples."""
    if len(r_series) == 0:
        return 0.0
    equity = np.insert(np.cumsum(r_series.values), 0, 0.0)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))

def print_yearly_report(trades_df: pd.DataFrame, label: str):
    """Print yearly breakdown for a single configuration."""
    print(f"\n=== {label} ===")
    if trades_df.empty:
        print("No trades generated.")
        return

    header = f"{'Year':<6} | {'Trades':<6} | {'Winrate':<8} | {'Exp (R)':<8} | {'Total R':<8} | {'Max DD (R)':<10}"
    print(header)
    print("-" * len(header))

    for year, grp in trades_df.groupby('year'):
        n = len(grp)
        wr = (grp['r_result'] > 0).mean() * 100
        exp_r = grp['r_result'].mean()
        tot_r = grp['r_result'].sum()
        mdd = calc_max_dd(grp['r_result'])
        print(f"{year:<6} | {n:<6} | {wr:6.2f}%  | {exp_r:+7.2f}R | {tot_r:+7.2f}R | -{mdd:8.2f}R")

    ov_n = len(trades_df)
    ov_wr = (trades_df['r_result'] > 0).mean() * 100
    ov_exp = trades_df['r_result'].mean()
    ov_tot = trades_df['r_result'].sum()
    ov_mdd = calc_max_dd(trades_df['r_result'])

    print("-" * len(header))
    print(f"{'TOTAL':<6} | {ov_n:<6} | {ov_wr:6.2f}%  | {ov_exp:+7.2f}R | {ov_tot:+7.2f}R | -{ov_mdd:8.2f}R")

if __name__ == "__main__":
    data = load_and_prepare_data(CSV_FILE)

    # Compare Full Range SL (1.0x) vs Mid-Range SL (0.5x)
    configs = [
        ("Full Range SL (1.0x) + Daily EMA50 + ATR Filter", 1.0),
        ("Mid-Range SL (0.5x)  + Daily EMA50 + ATR Filter", 0.5)
    ]

    summary_rows = []
    for title, sl_mult in configs:
        res_df = run_asian_breakout_backtest(data, sl_range_mult=sl_mult, rr_ratio=RR_RATIO)
        print_yearly_report(res_df, title)

        if not res_df.empty:
            summary_rows.append({
                'Strategy': f"SL={sl_mult}x (RR 1:{RR_RATIO})",
                'Trades': len(res_df),
                'Winrate_%': (res_df['r_result'] > 0).mean() * 100,
                'Exp_R': res_df['r_result'].mean(),
                'Total_R': res_df['r_result'].sum(),
                'Max_DD_R': -calc_max_dd(res_df['r_result'])
            })

    print("\n=== HEAD-TO-HEAD COMPARISON ===")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))