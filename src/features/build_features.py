import os
import pandas as pd
import numpy as np

def create_features(filepath_in, filepath_out):
    print(f"Loading raw data from {filepath_in}...")
    df = pd.read_csv(filepath_in)
    
    # In the real PJM dataset, the column is 'Datetime' and 'PJME_MW'
    # Our synthetic generator uses 'Datetime' and 'MW_Load'
    time_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
    target_col = [col for col in df.columns if col != time_col][0]
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    
    # 1. Time-series temporal features
    print("Engineering temporal features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    # 2. Lag features (Rolling window logic)
    # We want to know the demand 1 hour ago, 2 hours ago, and 24 hours ago (yesterday)
    print("Engineering lag and rolling window features...")
    df['lag_1h'] = df[target_col].shift(1)
    df['lag_2h'] = df[target_col].shift(2)
    df['lag_24h'] = df[target_col].shift(24) # exact same hour yesterday
    
    # Rolling averages over the past 24 hours to smooth out the noise
    df['rolling_mean_24h'] = df[target_col].shift(1).rolling(window=24).mean()
    df['rolling_std_24h'] = df[target_col].shift(1).rolling(window=24).std()
    
    # Drop NaNs created by shifts and rolling windows
    df = df.dropna()
    
    print(f"Feature engineering complete. Target variable: {target_col}")
    print(f"Total features created: {len(df.columns) - 1}")
    
    # Save processed dataframe
    os.makedirs(os.path.dirname(filepath_out), exist_ok=True)
    df.to_csv(filepath_out)
    print(f"Saved processed data to {filepath_out} ({len(df)} rows)")

if __name__ == "__main__":
    raw_path = "data/raw/PJME_hourly.csv"
    if not os.path.exists(raw_path):
        # Fallback to synthetic if download failed
        raw_path = "data/raw/Synthetic_Energy_Hourly.csv"
        
    create_features(raw_path, "data/processed/featured_data.csv")
