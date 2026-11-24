import pandas as pd
import numpy as np
from train_meta_model import (
    load_data,
    create_features,
    generate_primary_signals,
    apply_meta_labeling
)

if __name__ == "__main__":
    # 1. Load and prepare base data
    df = load_data("XAUUSD_M15.csv")
    df = create_features(df)
    df = generate_primary_signals(df)
    
    # 2. Apply session and volatility filters to primary_signal before looping (faster execution)
    valid_filter = (df['hour'] >= 1) & (df['hour'] <= 11) & (df['vol_regime'] < 1.1)
    df.loc[~valid_filter, 'primary_signal'] = 0
    
    # 3. Slice ONLY the first 85% (In-Sample) to prevent data leakage into Test Set
    in_sample_end = int(len(df) * 0.85)
    in_sample_df = df.iloc[:in_sample_end].copy()
    
    total_signals = (in_sample_df['primary_signal'] != 0).sum()
    print(f"Running Grid Search on {total_signals} filtered In-Sample signals...\n")
    
    # 4. Define Grid Search parameter space
    sl_list = [1.0, 1.5, 2.0]
    tp_list = [1.5, 2.0, 2.5, 3.0]
    max_bars_list = [16, 24, 32]
    
    results = []
    
    for sl in sl_list:
        for tp in tp_list:
            rr = tp / sl
            for mb in max_bars_list:
                # Apply Triple Barrier with current parameters
                temp_df = apply_meta_labeling(in_sample_df, tp_mult=tp, sl_mult=sl, max_bars=mb)
                
                # Evaluate only valid signal rows
                eval_df = temp_df[temp_df['primary_signal'] != 0].dropna(subset=['meta_target'])
                trades = len(eval_df)
                
                if trades == 0:
                    continue
                    
                winrate = eval_df['meta_target'].mean()
                # Conservative Expectancy: treats time-limit expiration (0) as a full -1R loss
                expectancy_r = (winrate * rr) - ((1.0 - winrate) * 1.0)
                
                results.append({
                    'sl_mult': sl,
                    'tp_mult': tp,
                    'RR': rr,
                    'max_bars': mb,
                    'trades': trades,
                    'winrate_%': winrate * 100,
                    'expectancy_R': expectancy_r
                })
                
    # 5. Sort by Expectancy (R) descending and display Top 15
    res_df = pd.DataFrame(results).sort_values(by='expectancy_R', ascending=False).reset_index(drop=True)
    
    print("=== Top 15 Parameter Combinations (Sorted by Expectancy R) ===")
    print(res_df.head(15).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))