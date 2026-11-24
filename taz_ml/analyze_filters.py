import pandas as pd
import numpy as np
from train_meta_model import (
    load_data,
    create_features,
    generate_primary_signals,
    apply_meta_labeling
)

def print_stats(df: pd.DataFrame, group_col: str, rr: float = 2.0):
    """Calculate trade count, winrate, and R-multiple expectancy per group."""
    stats = df.groupby(group_col, observed=False)['meta_target'].agg(
        trades='count',
        winrate=lambda x: x.mean() * 100
    )
    # Expectancy in R: (Winrate * RR) - (Lossrate * 1)
    stats['expectancy_R'] = ((stats['winrate'] / 100) * rr) - ((1 - (stats['winrate'] / 100)) * 1.0)
    stats['status'] = np.where(stats['expectancy_R'] > 0.10, 'KEEP', 
                      np.where(stats['expectancy_R'] > 0.0, 'MARGINAL', 'DROP'))
    
    print(f"\n=== Analysis by {group_col.upper()} (RR 1:{rr}) ===")
    print(stats.to_string(float_format=lambda x: f"{x:6.2f}"))

if __name__ == "__main__":
    # 1. Prepare data using existing pipeline
    df = load_data("XAUUSD_M15.csv")
    df = create_features(df)
    df = generate_primary_signals(df)
    df = apply_meta_labeling(df, tp_mult=2.0, sl_mult=1.0, max_bars=16)
    
    # 2. Filter only valid primary signals
    sig_df = df[df['primary_signal'] != 0].dropna(subset=['meta_target', 'vol_regime']).copy()
    
    # 3. Use ONLY the first 85% (Train + Val) to avoid peeking at Test Set
    in_sample_end = int(len(sig_df) * 0.85)
    in_sample_df = sig_df.iloc[:in_sample_end].copy()
    
    print(f"Analyzing In-Sample Signals: {len(in_sample_df)} trades")
    print(f"Overall In-Sample Winrate: {in_sample_df['meta_target'].mean()*100:.2f}%")
    
    # 4. Analyze by Hour (0 - 23)
    print_stats(in_sample_df, 'hour', rr=2.0)
    
    # 5. Analyze by Volatility Regime (ATR14 / ATR100)
    bins = [0, 0.7, 0.9, 1.1, 1.4, 2.0, np.inf]
    labels = ['<0.7 (Dead)', '0.7-0.9 (Low)', '0.9-1.1 (Normal)', '1.1-1.4 (Active)', '1.4-2.0 (High)', '>2.0 (Extreme)']
    in_sample_df['vol_bin'] = pd.cut(in_sample_df['vol_regime'], bins=bins, labels=labels)
    
    print_stats(in_sample_df, 'vol_bin', rr=2.0)