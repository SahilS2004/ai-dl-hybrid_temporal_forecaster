import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

from phase3_model import SynergisticForecaster, GMMRegimeDetector

def prepare_data(data_path, seq_len=24):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    X_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h']
    y_col = [c for c in df.columns if c not in X_cols][0]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(df[X_cols])
    y_scaled = scaler_y.fit_transform(df[[y_col]])
    
    # Pre-train GMM for regimes
    # We use 'rolling_std_24h' (index 8) to find high-volatility regimes
    detector = GMMRegimeDetector(n_components=3)
    detector.fit(X_scaled[:, 8].reshape(-1, 1))
    regime_labels = detector.gmm.predict(X_scaled[:, 8].reshape(-1, 1))
    
    xs, ys, rs = [], [], []
    for i in range(len(X_scaled) - seq_len):
        xs.append(X_scaled[i:i+seq_len])
        ys.append(y_scaled[i+seq_len])
        rs.append(regime_labels[i+seq_len])
        
    return (torch.FloatTensor(np.array(xs)), 
            torch.FloatTensor(np.array(ys)), 
            torch.LongTensor(np.array(rs)))

def train_phase3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Phase 3 Synergistic Model on {device}...")
    
    X, y, r = prepare_data("data/processed/featured_data.csv")
    dataset = TensorDataset(X, y, r)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = SynergisticForecaster(num_features=9).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_X, b_y, b_r in loader:
            b_X, b_y, b_r = b_X.to(device), b_y.to(device), b_r.to(device)
            
            optimizer.zero_grad()
            preds = model(b_X, b_r)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        
    os.makedirs("reports/advanced", exist_ok=True)
    torch.save(model.state_dict(), "reports/advanced/best_model.pth")
    print("✅ Phase 3 Model saved to reports/advanced/best_model.pth")

if __name__ == "__main__":
    train_phase3()
