import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "XAUUSD_M15.csv"
SPREAD = 0.20

ASIAN_START_HOUR = 0
ASIAN_END_HOUR = 6
LONDON_START_HOUR = 7
LONDON_END_HOUR = 11
EOD_EXIT_HOUR = 21

MIN_ATR_MULT = 0.20
MAX_ATR_MULT = 0.80
DAILY_EMA_PERIOD = 50

SL_RANGE_MULT = 1.0
RR_RATIO = 1.5

# ==========================================
# 1. LOAD DATA & COMPUTE BASE INDICATORS
# ==========================================
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load M15 OHLCV data, M15 volume baseline, and shifted Daily ATR/EMA."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['year'] = df['time'].dt.year
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek

    # M15 relative volume (20-bar moving average)
    df['vol_ma20'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)

    # Aggregate M15 bars into Daily OHLC
    daily = df.groupby('date').agg(
        d_open=('open', 'first'),
        d_high=('high', 'max'),
        d_low=('low', 'min'),
        d_close=('close', 'last')
    ).reset_index()

    # Daily True Range & ATR(14)
    d_hl = daily['d_high'] - daily['d_low']
    d_hc = np.abs(daily['d_high'] - daily['d_close'].shift(1))
    d_lc = np.abs(daily['d_low'] - daily['d_close'].shift(1))
    daily['d_tr'] = pd.concat([d_hl, d_hc, d_lc], axis=1).max(axis=1)
    daily['d_atr14'] = daily['d_tr'].rolling(window=14).mean()
    daily['d_ema'] = daily['d_close'].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()

    # Shift by 1 day to strictly avoid lookahead bias
    daily['prev_d_atr'] = daily['d_atr14'].shift(1)
    daily['prev_d_close'] = daily['d_close'].shift(1)
    daily['prev_d_ema'] = daily['d_ema'].shift(1)
    daily['prev_d_trend'] = np.where(daily['prev_d_close'] > daily['prev_d_ema'], 1, -1)
    daily.loc[daily['prev_d_atr'].isna(), 'prev_d_trend'] = 0

    cols_to_merge = ['date', 'prev_d_atr', 'prev_d_close', 'prev_d_ema', 'prev_d_trend']
    df = df.merge(daily[cols_to_merge], on='date', how='left')
    return df

# ==========================================
# 2. GENERATE BREAKOUT DATASET + FEATURES
# ==========================================
def build_breakout_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Asian Range Breakout setups, breakout-specific features, and net R outcomes."""
    records = []

    for date, day_df in df.groupby('date'):
        prev_atr = day_df['prev_d_atr'].iloc[0]
        prev_trend = day_df['prev_d_trend'].iloc[0]
        prev_d_close = day_df['prev_d_close'].iloc[0]
        prev_d_ema = day_df['prev_d_ema'].iloc[0]

        if pd.isna(prev_atr) or prev_atr <= 0 or prev_trend == 0:
            continue

        # Asian Session Range
        asian_df = day_df[(day_df['hour'] >= ASIAN_START_HOUR) & (day_df['hour'] <= ASIAN_END_HOUR)]
        if len(asian_df) < 8:
            continue

        asian_high = asian_df['high'].max()
        asian_low = asian_df['low'].min()
        asian_range = asian_high - asian_low

        if asian_range < (prev_atr * MIN_ATR_MULT) or asian_range > (prev_atr * MAX_ATR_MULT):
            continue

        # London Breakout Window
        london_df = day_df[(day_df['hour'] >= LONDON_START_HOUR) & (day_df['hour'] <= LONDON_END_HOUR)]
        if london_df.empty:
            continue

        entry_idx = None
        direction = 0

        for idx, row in london_df.iterrows():
            if prev_trend == 1 and row['close'] > asian_high:
                entry_idx = idx
                direction = 1
                break
            elif prev_trend == -1 and row['close'] < asian_low:
                entry_idx = idx
                direction = -1
                break

        if entry_idx is None:
            continue

        entry_bar = day_df.loc[entry_idx]
        sl_dist = asian_range * SL_RANGE_MULT
        tp_dist = sl_dist * RR_RATIO

        # Simulate trade path with real spread deduction
        if direction == 1:
            entry_price = entry_bar['close'] + SPREAD
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            entry_price = entry_bar['close']
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        future_bars = day_df[(day_df.index > entry_idx) & (day_df['hour'] <= EOD_EXIT_HOUR)]
        if future_bars.empty:
            continue

        r_result = None
        for _, f_row in future_bars.iterrows():
            if direction == 1:
                if f_row['low'] <= sl_price:
                    r_result = -1.0
                    break
                if f_row['high'] >= tp_price:
                    r_result = RR_RATIO
                    break
            else:
                if (f_row['high'] + SPREAD) >= sl_price:
                    r_result = -1.0
                    break
                if (f_row['low'] + SPREAD) <= tp_price:
                    r_result = RR_RATIO
                    break

        if r_result is None:
            last_close = future_bars.iloc[-1]['close']
            pnl = (last_close - entry_price) if direction == 1 else (entry_price - (last_close + SPREAD))
            r_result = pnl / sl_dist

        # --- Breakout-Specific Feature Engineering ---
        candle_range = entry_bar['high'] - entry_bar['low']
        candle_body = abs(entry_bar['close'] - entry_bar['open'])
        candle_body_ratio = (candle_body / candle_range) if candle_range > 0 else 0.0
        candle_size_norm = candle_range / asian_range

        if direction == 1:
            # Distance closed above Asian High (measures over-extension / chasing)
            breakout_dist_norm = (entry_bar['close'] - asian_high) / asian_range
            # Upper wick against Buy direction (selling pressure)
            wick_against = entry_bar['high'] - max(entry_bar['open'], entry_bar['close'])
        else:
            # Distance closed below Asian Low
            breakout_dist_norm = (asian_low - entry_bar['close']) / asian_range
            # Lower wick against Sell direction (buying pressure)
            wick_against = min(entry_bar['open'], entry_bar['close']) - entry_bar['low']

        wick_against_ratio = (wick_against / candle_range) if candle_range > 0 else 0.0
        range_atr_ratio = asian_range / prev_atr
        dist_d_ema50_norm = abs(prev_d_close - prev_d_ema) / prev_atr

        records.append({
            'time': entry_bar['time'],
            'year': entry_bar['year'],
            'direction': direction,
            'range_atr_ratio': range_atr_ratio,
            'breakout_dist_norm': breakout_dist_norm,
            'candle_body_ratio': candle_body_ratio,
            'candle_size_norm': candle_size_norm,
            'wick_against_ratio': wick_against_ratio,
            'dist_d_ema50_norm': dist_d_ema50_norm,
            'vol_ratio': entry_bar['vol_ratio'],
            'entry_hour': entry_bar['hour'],
            'dayofweek': entry_bar['dayofweek'],
            'r_result': r_result,
            'meta_target': 1 if r_result > 0 else 0
        })

    return pd.DataFrame(records)

# ==========================================
# 3. METRICS & 5-FOLD WALK-FORWARD VALIDATION
# ==========================================
def calc_metrics(r_arr: np.ndarray):
    """Calculate Total R, Max Drawdown (R), Winrate (%), and Expectancy (R)."""
    if len(r_arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    equity = np.insert(np.cumsum(r_arr), 0, 0.0)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))
    total_r = float(np.sum(r_arr))
    wr = float(np.mean(r_arr > 0) * 100.0)
    exp_r = float(np.mean(r_arr))
    return total_r, max_dd, wr, exp_r

def run_walk_forward(dataset: pd.DataFrame, feature_cols: list, n_folds: int = 5, keep_top_pct: float = 65.0):
    """Run Expanding Window 5-Fold Walk-Forward Validation comparing Baseline vs AI-Filtered."""
    df_clean = dataset.dropna(subset=feature_cols + ['meta_target']).reset_index(drop=True)
    n = len(df_clean)
    chunk_size = n // (n_folds + 1)

    print(f"=== Asian Breakout Meta-Labeling Walk-Forward ({n_folds} Folds | Keep Top {keep_top_pct:.0f}%) ===")
    print(f"Total Valid Setups: {n} (~{chunk_size} setups per fold)\n")

    header = (
        f"{'Fold':<5} | {'Test Period':<23} | {'Base R':<8} | {'AI Trd':<6} | "
        f"{'AI WR':<7} | {'AI Exp':<8} | {'AI Total':<8} | {'AI MaxDD':<9}"
    )
    print(header)
    print("-" * len(header))

    all_base_r = []
    all_ai_r = []
    last_model = None

    for fold in range(1, n_folds + 1):
        train_val_end = fold * chunk_size
        test_start = train_val_end
        test_end = (fold + 1) * chunk_size if fold < n_folds else n

        train_val_df = df_clean.iloc[:train_val_end]
        test_df = df_clean.iloc[test_start:test_end]

        # Split 80% Train / 20% Validation
        val_split = int(len(train_val_df) * 0.80)
        train_df = train_val_df.iloc[:val_split]
        val_df = train_val_df.iloc[val_split:]

        X_train, y_train = train_df[feature_cols], train_df['meta_target']
        X_val, y_val = val_df[feature_cols], val_df['meta_target']
        X_test = test_df[feature_cols]

        # Highly regularized LightGBM for small-to-medium trade counts
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=7,
            max_depth=3,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        last_model = model

        # Dynamic threshold from Validation Set to drop the worst (100 - keep_top_pct)% setups
        val_probs = model.predict_proba(X_val)[:, 1]
        dyn_thresh = np.percentile(val_probs, 100.0 - keep_top_pct)

        test_probs = model.predict_proba(X_test)[:, 1]
        selected_mask = (test_probs >= dyn_thresh)

        base_r_arr = test_df['r_result'].values
        ai_r_arr = test_df['r_result'].values[selected_mask]

        all_base_r.extend(base_r_arr)
        all_ai_r.extend(ai_r_arr)

        base_tot, _, _, _ = calc_metrics(base_r_arr)
        ai_tot, ai_mdd, ai_wr, ai_exp = calc_metrics(ai_r_arr)

        period_str = f"{test_df['time'].iloc[0].strftime('%Y-%m-%d')} to {test_df['time'].iloc[-1].strftime('%Y-%m-%d')}"
        print(
            f"F{fold:<4} | {period_str:<23} | {base_tot:+7.2f}R | {len(ai_r_arr):<6} | "
            f"{ai_wr:5.2f}%  | {ai_exp:+7.2f}R | {ai_tot:+7.2f}R | -{ai_mdd:7.2f}R"
        )

    # Overall Summary Comparison
    b_tot, b_mdd, b_wr, b_exp = calc_metrics(np.array(all_base_r))
    a_tot, a_mdd, a_wr, a_exp = calc_metrics(np.array(all_ai_r))

    print("-" * len(header))
    print(f"BASE  | {'All 5 OOS Folds (No AI)':<23} | {b_tot:+7.2f}R | {len(all_base_r):<6} | {b_wr:5.2f}%  | {b_exp:+7.2f}R | {b_tot:+7.2f}R | -{b_mdd:7.2f}R")
    print(f"AI    | {'All 5 OOS Folds (AI)':<23} | {b_tot:+7.2f}R | {len(all_ai_r):<6} | {a_wr:5.2f}%  | {a_exp:+7.2f}R | {a_tot:+7.2f}R | -{a_mdd:7.2f}R")

    # Feature Importance from Final Fold
    imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Top 5 Breakout Feature Importance (Final Fold) ===")
    print(imp.head(5))

    return last_model

if __name__ == "__main__":
    raw_df = load_and_prepare_data(CSV_FILE)
    breakout_df = build_breakout_dataset(raw_df)

    FEATURES = [
        'direction', 'range_atr_ratio', 'breakout_dist_norm',
        'candle_body_ratio', 'candle_size_norm', 'wick_against_ratio',
        'dist_d_ema50_norm', 'vol_ratio', 'entry_hour', 'dayofweek'
    ]

    final_model = run_walk_forward(breakout_df, FEATURES, n_folds=5, keep_top_pct=65.0)
    joblib.dump(final_model, "meta_asian_breakout.joblib")
    print("\nSaved final model as meta_asian_breakout.joblib")