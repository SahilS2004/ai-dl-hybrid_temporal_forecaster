import os
import urllib.request
import pandas as pd
import numpy as np

def generate_synthetic_energy_data(filepath, n_days=365*4):
    """
    Fallback: Generates a highly realistic Hourly Energy Load dataset
    with daily/weekly/yearly seasonality, and explicit regime shifts 
    to satisfy the project requirements.
    """
    print("Generating a synthetic complex energy load dataset...")
    
    dates = pd.date_range(start="2018-01-01", periods=n_days*24, freq="h")
    
    # Base load
    load = pd.Series(30000 + np.sin(dates.dayofyear * (2 * np.pi / 365.25)) * 5000)  # Yearly
    load += pd.Series(np.sin(dates.hour * (2 * np.pi / 24)) * 3000)                    # Daily
    
    # Weekend effect: lower load on weekends
    weekend_mask = dates.dayofweek >= 5
    load[weekend_mask] *= 0.85
    
    # 🌟 REGIME SHIFT 1: The "Pandemic Lockdown" (Drop in structural load starting March 2020)
    lockdown_mask = (dates >= "2020-03-15") & (dates <= "2020-09-01")
    load[lockdown_mask] *= 0.70  
    
    # 🌟 REGIME SHIFT 2: The "Extreme Heat Wave" (Sudden spike in variance/load)
    heatwave_mask = (dates >= "2021-07-01") & (dates <= "2021-07-15")
    load[heatwave_mask] += 12000
    
    # Noise
    noise = np.random.normal(0, 1000, len(load))
    load += noise
    
    df = pd.DataFrame({"Datetime": dates, "MW_Load": load})
    df.to_csv(filepath, index=False)
    print(f"Synthetic dataset saved to {filepath} ({len(df)} rows)")

def main():
    os.makedirs("data/raw", exist_ok=True)
    filepath = "data/raw/PJME_hourly.csv"
    
    url = "https://raw.githubusercontent.com/robmed/pm-time-series/master/data/PJME_hourly.csv"
    
    try:
        print(f"Attempting to download genuine PJM Hourly dataset from {url}...")
        urllib.request.urlretrieve(url, filepath)
        
        # Validate CSV format
        df = pd.read_csv(filepath)
        if len(df) < 1000:
            raise ValueError("Downloaded file is too small to be the right dataset.")
            
        print(f"Successfully downloaded realistic real-world dataset to {filepath}! ({len(df)} rows)")
        
    except Exception as e:
        print(f"Failed to download real dataset: {e}")
        # Generate the synthetic data as a reliable fallback
        generate_synthetic_energy_data("data/raw/Synthetic_Energy_Hourly.csv")

if __name__ == "__main__":
    main()
