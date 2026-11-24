import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data exported from MT4 CSV."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary features normalized by ATR to handle price regime shifts."""
    df = df.copy()
    
    # Calculate True Range and ATR(14)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Trend Features: Distance from EMAs normalized by ATR
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()
    
    df['dist_ema20'] = (df['close'] - ema_20) / df['atr_14']
    df['dist_ema50'] = (df['close'] - ema_50) / df['atr_14']
    df['dist_ema200'] = (df['close'] - ema_200) / df['atr_14']
    df['ema_slope'] = (ema_20 - ema_50) / df['atr_14']
    
    # Momentum Feature: MACD Histogram normalized by ATR
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist_norm'] = (macd_line - signal_line) / df['atr_14']
    
    # Price Action Features: Candle body and wicks normalized by ATR
    df['body_norm'] = (df['close'] - df['open']) / df['atr_14']
    df['upper_wick_norm'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr_14']
    df['lower_wick_norm'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr_14']
    
    # Volume Anomaly: Relative tick volume vs 20-bar moving average
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Time Features: Trading session context
    df['hour'] = df['time'].dt.hour
    df['dayofweek'] = df['time'].dt.dayofweek
    
    return df

# ==========================================
# 3. TRIPLE BARRIER LABELING
# ==========================================
def apply_triple_barrier(df: pd.DataFrame, tp_mult: float = 2.0, sl_mult: float = 1.0, max_bars: int = 12) -> pd.DataFrame:
    """
    Generate multiclass target labels:
    0 = Hold / No Trade (SL hit or time limit expired)
    1 = Buy Signal (Buy TP hit before Buy SL)
    2 = Sell Signal (Sell TP hit before Sell SL)
    """
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = df['atr_14'].values
    
    for i in range(len(df) - max_bars):
        if np.isnan(atr_arr[i]) or atr_arr[i] == 0:
            continue
            
        entry_price = close_arr[i]
        tp_dist = atr_arr[i] * tp_mult
        sl_dist = atr_arr[i] * sl_mult
        
        buy_tp = entry_price + tp_dist
        buy_sl = entry_price - sl_dist
        sell_tp = entry_price - tp_dist
        sell_sl = entry_price + sl_dist
        
        buy_win = False
        sell_win = False
        
        # Check Buy outcome first (Conservative: check SL before TP on the same bar)
        for j in range(1, max_bars + 1):
            if low_arr[i + j] <= buy_sl:
                break
            if high_arr[i + j] >= buy_tp:
                buy_win = True
                break
                
        # Check Sell outcome (Conservative: check SL before TP on the same bar)
        for j in range(1, max_bars + 1):
            if high_arr[i + j] >= sell_sl:
                break
            if low_arr[i + j] <= sell_tp:
                sell_win = True
                break
                
        # Assign label only if direction is unambiguous
        if buy_win and not sell_win:
            labels[i] = 1  # BUY
        elif sell_win and not buy_win:
            labels[i] = 2  # SELL
        else:
            labels[i] = 0  # HOLD
            
    df['target'] = labels
    return df

# ==========================================
# 4. TIME-SERIES SPLIT & TRAINING
# ==========================================
def train_and_evaluate(df: pd.DataFrame, feature_cols: list):
    """Split data chronologically and train LightGBM classifier."""
    # Drop NaN rows created by indicators and trailing max_bars
    df_clean = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    # Chronological Split: 70% Train, 15% Validation (Early Stopping), 15% Out-of-Sample Test
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]
    
    X_train, y_train = train_df[feature_cols], train_df['target']
    X_val, y_val = val_df[feature_cols], val_df['target']
    X_test, y_test = test_df[feature_cols], test_df['target']
    
    # Initialize LightGBM Classifier
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Train with Early Stopping on Validation Set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate on Out-of-Sample Test Set
    y_pred = model.predict(X_test)
    print("=== Out-of-Sample Test Report ===")
    print(classification_report(y_test, y_pred, target_names=['HOLD (0)', 'BUY (1)', 'SELL (2)']))
    
    # Check High-Confidence Signals (e.g., Probability >= 0.65)
    probs = model.predict_proba(X_test)
    high_conf_buy = (probs[:, 1] >= 0.65)
    high_conf_sell = (probs[:, 2] >= 0.65)
    print(f"High-confidence BUY signals (>=65%): {high_conf_buy.sum()}")
    print(f"High-confidence SELL signals (>=65%): {high_conf_sell.sum()}")
    
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load historical CSV (Columns: time, open, high, low, close, volume)
    data = load_data("XAUUSD_M15.csv")
    
    # 2. Generate Features
    data = create_features(data)
    
    # 3. Generate Triple Barrier Labels (RR 1:2, Max holding 12 bars = 3 hours on M15)
    data = apply_triple_barrier(data, tp_mult=2.0, sl_mult=1.0, max_bars=12)
    
    # Define feature columns used for training
    FEATURES = [
        'dist_ema20', 'dist_ema50', 'dist_ema200', 'ema_slope',
        'macd_hist_norm', 'body_norm', 'upper_wick_norm', 'lower_wick_norm',
        'vol_ratio', 'hour', 'dayofweek'
    ]
    
    # 4. Train and Save Model
    trained_model = train_and_evaluate(data, FEATURES)
    joblib.dump(trained_model, "lgbm_xauusd_m15.joblib")
    print("Model saved successfully as lgbm_xauusd_m15.joblib")