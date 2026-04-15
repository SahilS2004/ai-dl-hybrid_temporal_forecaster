import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import os
import sys

sys.path.append('src/models')
from advanced_hybrid_model import AdvancedHybridForecaster

def generate_visuals(data_path="data/processed/featured_data.csv", model_path="reports/advanced/best_model.pth"):
    print("Generating Level-10 Technical Validation Visuals...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    X_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h']
    
    # GMM
    gmm_scaler = StandardScaler()
    gmm_input = gmm_scaler.fit_transform(df[['rolling_mean_24h', 'rolling_std_24h']])
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(gmm_input)
    gmm_probs = gmm.predict_proba(gmm_input)
    
    # Seq
    seq_length = 24
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(df[X_cols])
    R_data = gmm_probs
    
    X_seq, R_seq = [], []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
        R_seq.append(R_data[i:i+seq_length])
    
    X_test = torch.FloatTensor(X_seq[-100:])
    R_test = torch.FloatTensor(R_seq[-100:])
    
    model = AdvancedHybridForecaster(num_features=len(X_cols))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        _ = model(X_test[0:1], R_test[0:1])
        attn = model.attn_map[0].numpy()
        
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis')
    plt.title("Level-10 Tech Validation: Regime-Aware Attention Map")
    plt.xlabel("Temporal Lags (T-24 to T-1)")
    plt.ylabel("Output Feature Weights")
    plt.colorbar()
    os.makedirs("reports/advanced", exist_ok=True)
    plt.savefig("reports/advanced/attention_map.png")
    print("Attention map saved.")

if __name__ == "__main__":
    generate_visuals()
