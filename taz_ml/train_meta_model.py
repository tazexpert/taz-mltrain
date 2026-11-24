import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data exported from MT4 CSV (Bid prices)."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary features normalized by ATR."""
    df = df.copy()
    
    # Calculate True Range and ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_100'] = tr.rolling(window=100).mean()
    
    # Volatility Regime: Current volatility vs long-term average
    df['vol_regime'] = df['atr_14'] / df['atr_100']
    
    # EMAs for Primary Strategy and Features
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Bollinger Bands (20, 2.0)
    bb_mid = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = bb_mid + (2.0 * bb_std)
    df['bb_lower'] = bb_mid - (2.0 * bb_std)
    df['bb_width_norm'] = (df['bb_upper'] - df['bb_lower']) / df['atr_14']
    
    # Normalized Trend Distances
    df['dist_ema20'] = (df['close'] - df['ema_20']) / df['atr_14']
    df['dist_ema50'] = (df['close'] - df['ema_50']) / df['atr_14']
    df['dist_ema200'] = (df['close'] - df['ema_200']) / df['atr_14']
    df['ema_spread'] = (df['ema_20'] - df['ema_50']) / df['atr_14']
    
    # Momentum: MACD Histogram normalized by ATR
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist_norm'] = (macd_line - signal_line) / df['atr_14']
    
    # Price Action Features: Candle body and wicks normalized by ATR
    df['body_norm'] = (df['close'] - df['open']) / df['atr_14']
    df['upper_wick_norm'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr_14']
    df['lower_wick_norm'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr_14']
    
    # Volume Anomaly
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Time Features
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    
    return df

# ==========================================
# 3. PRIMARY STRATEGY (RELAXED FOR MORE SAMPLES)
# ==========================================
def generate_primary_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primary Strategy: Bollinger Band Rejection + EMA200 Trend.
    Relaxed hour, vol_regime, and vol_ratio so LightGBM can learn optimal filtering.
    Returns: 1 (Buy Setup), -1 (Sell Setup), 0 (No Setup)
    """
    df = df.copy()
    
    # Exclude only the extreme kill-zone (15:00-19:00) where spreads/whipsaws dominate
    not_killzone = ~((df['hour'] >= 15) & (df['hour'] <= 19))
    
    # Buy Condition: Above EMA200 + Lower BB Rejection + Bullish Candle
    buy_cond = (
        not_killzone &
        (df['close'] > df['ema_200']) &
        (df['low'] <= df['bb_lower']) &
        (df['close'] > df['bb_lower']) &
        (df['close'] > df['open'])
    )
    
    # Sell Condition: Below EMA200 + Upper BB Rejection + Bearish Candle
    sell_cond = (
        not_killzone &
        (df['close'] < df['ema_200']) &
        (df['high'] >= df['bb_upper']) &
        (df['close'] < df['bb_upper']) &
        (df['close'] < df['open'])
    )
    
    df['primary_signal'] = 0
    df.loc[buy_cond, 'primary_signal'] = 1
    df.loc[sell_cond, 'primary_signal'] = -1
    
    return df

# ==========================================
# 4. META-LABELING WITH REAL SPREAD DEDUCTION
# ==========================================
def apply_meta_labeling(
    df: pd.DataFrame,
    tp_mult: float = 2.5,
    sl_mult: float = 1.5,
    max_bars: int = 32,
    spread: float = 0.20
) -> pd.DataFrame:
    """
    Evaluate primary signals with realistic Bid/Ask spread deduction ($0.20 default for XAUUSD).
    Target: 1 if trade hits TP before SL within max_bars, else 0.
    """
    df = df.copy()
    meta_targets = np.full(len(df), np.nan)
    
    signals = df['primary_signal'].values
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr_14'].values
    
    for i in range(len(df) - max_bars):
        sig = signals[i]
        if sig == 0 or np.isnan(atr_arr[i]) or atr_arr[i] == 0:
            continue
            
        tp_dist = atr_arr[i] * tp_mult
        sl_dist = atr_arr[i] * sl_mult
        
        win = 0
        if sig == 1:  # BUY: Enter at Ask (close + spread), exit at Bid (high/low)
            entry_ask = close_arr[i] + spread
            tp_price = entry_ask + tp_dist
            sl_price = entry_ask - sl_dist
            for j in range(1, max_bars + 1):
                if low_arr[i + j] <= sl_price:  # Conservative: check SL first
                    break
                if high_arr[i + j] >= tp_price:
                    win = 1
                    break
                    
        elif sig == -1:  # SELL: Enter at Bid (close), exit at Ask (high/low + spread)
            entry_bid = close_arr[i]
            tp_price = entry_bid - tp_dist
            sl_price = entry_bid + sl_dist
            for j in range(1, max_bars + 1):
                if (high_arr[i + j] + spread) >= sl_price:  # Conservative: check SL first
                    break
                if (low_arr[i + j] + spread) <= tp_price:
                    win = 1
                    break
                    
        meta_targets[i] = win
        
    df['meta_target'] = meta_targets
    return df

# ==========================================
# 5. TRAIN BINARY LIGHTGBM (MEDIUM DATASET TUNING)
# ==========================================
def train_meta_model(df: pd.DataFrame, feature_cols: list, rr: float):
    """Filter signal rows, split chronologically, and train regularized Binary LightGBM."""
    signal_df = df[df['primary_signal'] != 0].dropna(subset=feature_cols + ['meta_target']).reset_index(drop=True)
    signal_df['meta_target'] = signal_df['meta_target'].astype(int)
    
    print(f"Total Primary Signals found: {len(signal_df)}")
    
    # Chronological Split: 70% Train, 15% Validation, 15% Out-of-Sample Test
    n = len(signal_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = signal_df.iloc[:train_end]
    val_df = signal_df.iloc[train_end:val_end]
    test_df = signal_df.iloc[val_end:]
    
    X_train, y_train = train_df[feature_cols], train_df['meta_target']
    X_val, y_val = val_df[feature_cols], val_df['meta_target']
    X_test, y_test = test_df[feature_cols], test_df['meta_target']
    
    # Tuned for medium-sized tabular dataset (hundreds to low thousands of trades)
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=10,        # Prevent overfitting on medium sample size
        max_depth=4,
        min_child_samples=15, # Allow finer splits than default 20-30
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,        # L1 regularization
        reg_lambda=1.0,       # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate Baseline vs AI-Filtered (Net of Spread)
    baseline_wr = y_test.mean()
    baseline_exp = (baseline_wr * rr) - ((1.0 - baseline_wr) * 1.0)
    
    print(f"\n=== Out-of-Sample Test Set ({len(y_test)} trades, Net of Spread) ===")
    print(f"Baseline (No AI) -> Winrate: {baseline_wr*100:.2f}% | Expectancy: {baseline_exp:+.2f}R\n")
    
    probs = model.predict_proba(X_test)[:, 1]
    
    print(f"{'Threshold':<12} | {'Trades':<8} | {'Winrate':<10} | {'Expectancy (R)':<15}")
    print("-" * 55)
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55]:
        selected = (probs >= thresh)
        count = selected.sum()
        if count > 0:
            wr = y_test[selected].mean()
            exp_r = (wr * rr) - ((1.0 - wr) * 1.0)
            print(f">= {thresh:<9.2f} | {count:<8} | {wr*100:6.2f}%   | {exp_r:+.2f}R")
        else:
            print(f">= {thresh:<9.2f} | {0:<8} | {'N/A':<10} | N/A")
            
    # Display Top 5 Most Important Features
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Top 5 Feature Importance ===")
    print(importance.head(5))
    
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    TP_MULT = 2.5
    SL_MULT = 1.5
    RR_RATIO = TP_MULT / SL_MULT
    
    data = load_data("XAUUSD_M15.csv")
    data = create_features(data)
    data = generate_primary_signals(data)
    
    # Apply Triple Barrier with $0.20 gold spread deducted
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
    
    trained_meta_model = train_meta_model(data, FEATURES, rr=RR_RATIO)
    joblib.dump(trained_meta_model, "meta_lgbm_xauusd.joblib")
    print("\nMeta-model saved as meta_lgbm_xauusd.joblib")