import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from train_meta_model import (
    load_data,
    create_features,
    generate_primary_signals,
    apply_meta_labeling
)

def calculate_r_metrics(outcomes: np.ndarray, rr: float):
    """Calculate Total R, Max Drawdown (in R), Winrate, and Expectancy from binary outcomes."""
    if len(outcomes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Convert binary outcomes to R-multiples (+RR for win, -1.0 for loss)
    r_returns = np.where(outcomes == 1, rr, -1.0)
    equity_curve = np.cumsum(r_returns)
    
    # Prepend 0.0 so drawdown calculation starts from initial equity (0R)
    equity_with_start = np.insert(equity_curve, 0, 0.0)
    running_peak = np.maximum.accumulate(equity_with_start)
    drawdowns = running_peak - equity_with_start
    
    total_r = equity_curve[-1]
    max_dd_r = np.max(drawdowns)
    winrate = np.mean(outcomes) * 100.0
    expectancy_r = np.mean(r_returns)
    
    return total_r, max_dd_r, winrate, expectancy_r

def run_sliding_walk_forward(
    df: pd.DataFrame,
    feature_cols: list,
    rr: float,
    n_folds: int = 5,
    window_chunks: int = 2,
    top_pct: float = 25.0
):
    """
    Run Sliding Window Walk-Forward Validation with Dynamic Threshold (Top N% from Validation Set).
    """
    signal_df = df[df['primary_signal'] != 0].dropna(subset=feature_cols + ['meta_target']).reset_index(drop=True)
    signal_df['meta_target'] = signal_df['meta_target'].astype(int)
    
    # Divide data into (n_folds + window_chunks) equal chronological chunks
    total_samples = len(signal_df)
    total_chunks = n_folds + window_chunks
    chunk_size = total_samples // total_chunks
    
    print(f"=== Sliding Window Walk-Forward ({n_folds} Folds | Window: {window_chunks} chunks | Top {top_pct:.0f}% | RR 1:{rr:.2f}) ===")
    print(f"Total Filtered Signals: {total_samples} (~{chunk_size} signals/chunk, ~{chunk_size * window_chunks} train+val/fold)\n")
    
    header = (
        f"{'Fold':<5} | {'Test Period':<23} | {'Thresh':<6} | {'Trades':<6} | "
        f"{'Winrate':<8} | {'Exp (R)':<8} | {'Total R':<8} | {'Max DD (R)':<10}"
    )
    print(header)
    print("-" * len(header))
    
    all_oos_outcomes = []
    
    for fold in range(1, n_folds + 1):
        # Sliding training window: fixed length of 'window_chunks' immediately before test fold
        train_val_start = (fold - 1) * chunk_size
        train_val_end = (fold - 1 + window_chunks) * chunk_size
        test_start = train_val_end
        test_end = (fold + window_chunks) * chunk_size if fold < n_folds else total_samples
        
        train_val_df = signal_df.iloc[train_val_start:train_val_end]
        test_df = signal_df.iloc[test_start:test_end]
        
        # Split 80% Train / 20% Validation (for Early Stopping and Dynamic Threshold calibration)
        val_split = int(len(train_val_df) * 0.80)
        train_df = train_val_df.iloc[:val_split]
        val_df = train_val_df.iloc[val_split:]
        
        X_train, y_train = train_df[feature_cols], train_df['meta_target']
        X_val, y_val = val_df[feature_cols], val_df['meta_target']
        X_test, y_test = test_df[feature_cols], test_df['meta_target']
        
        # Train LightGBM model for current sliding window
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=10,
            max_depth=4,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Compute Dynamic Threshold from Validation Set (e.g., 75th percentile for Top 25%)
        val_probs = model.predict_proba(X_val)[:, 1]
        dyn_thresh = np.percentile(val_probs, 100.0 - top_pct)
        
        # Predict on Out-of-Sample Test Fold using the Dynamic Threshold
        test_probs = model.predict_proba(X_test)[:, 1]
        selected_mask = (test_probs >= dyn_thresh)
        fold_outcomes = y_test.values[selected_mask]
        
        all_oos_outcomes.extend(fold_outcomes)
        
        # Calculate fold metrics
        total_r, max_dd_r, wr, exp_r = calculate_r_metrics(fold_outcomes, rr)
        start_str = test_df['time'].iloc[0].strftime('%Y-%m-%d')
        end_str = test_df['time'].iloc[-1].strftime('%Y-%m-%d')
        period_str = f"{start_str} to {end_str}"
        
        print(
            f"F{fold:<4} | {period_str:<23} | {dyn_thresh:6.3f} | {len(fold_outcomes):<6} | "
            f"{wr:6.2f}%  | {exp_r:+7.2f}R | {total_r:+7.2f}R | -{max_dd_r:8.2f}R"
        )
        
    # Calculate Overall Continuous Out-of-Sample Performance
    all_oos_arr = np.array(all_oos_outcomes)
    ov_total_r, ov_max_dd_r, ov_wr, ov_exp_r = calculate_r_metrics(all_oos_arr, rr)
    
    print("-" * len(header))
    print(
        f"{'TOTAL':<5} | {'All 5 OOS Folds':<23} | {'Dyn':<6} | {len(all_oos_arr):<6} | "
        f"{ov_wr:6.2f}%  | {ov_exp_r:+7.2f}R | {ov_total_r:+7.2f}R | -{ov_max_dd_r:8.2f}R"
    )

if __name__ == "__main__":
    TP_MULT = 2.5
    SL_MULT = 1.5
    RR_RATIO = TP_MULT / SL_MULT
    
    data = load_data("XAUUSD_M15.csv")
    data = create_features(data)
    data = generate_primary_signals(data)
    data = apply_meta_labeling(
        data,
        tp_mult=TP_MULT,
        sl_mult=SL_MULT,
        max_bars=32,
        spread=0.20
    )
    
    FEATURES = [
        'primary_signal', 'vol_regime', 'bb_width_norm', 'dist_ema20', 'dist_ema50',
        'dist_ema200', 'ema_spread', 'macd_hist_norm', 'body_norm',
        'upper_wick_norm', 'lower_wick_norm', 'vol_ratio', 'hour', 'dayofweek'
    ]
    
    run_sliding_walk_forward(
        data,
        FEATURES,
        rr=RR_RATIO,
        n_folds=5,
        window_chunks=2,
        top_pct=25.0
    )