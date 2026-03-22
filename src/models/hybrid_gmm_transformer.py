import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy

# ---------------------------------------------------------
# Sequence Generation Component
# ---------------------------------------------------------
def create_sequences(X, y, seq_length=24):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

# ---------------------------------------------------------
# PyTorch Deep Learning Architecture
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        src = src.transpose(0, 1) 
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_layer(output[-1, :, :])

# ---------------------------------------------------------
# Hybrid Pipeline Strategy
# ---------------------------------------------------------
def train_and_evaluate_hybrid(data_path, results_dir="reports"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    train_size = 8760
    test_size = 2160
    seq_length = 24
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size - seq_length : train_size + test_size].copy()
    
    # 🌟 CORE HYBRID INNOVATION: Gaussian Mixture Model (GMM)
    # We train the probabilistic model ONLY on rolling variance to detect regimes
    print("Training GMM (Probabilistic Model) to identify Extreme Regimes...")
    gmm_features = ['rolling_mean_24h', 'rolling_std_24h']
    
    gmm_scaler = StandardScaler()
    train_gmm_scaled = gmm_scaler.fit_transform(train_df[gmm_features])
    test_gmm_scaled = gmm_scaler.transform(test_df[gmm_features])
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(train_gmm_scaled)
    
    # Extract probability distributions for explicit Deep Learning attention
    train_probs = gmm.predict_proba(train_gmm_scaled)
    test_probs = gmm.predict_proba(test_gmm_scaled)
    
    train_df['prob_state_0'] = train_probs[:, 0]
    train_df['prob_state_1'] = train_probs[:, 1]
    test_df['prob_state_0'] = test_probs[:, 0]
    test_df['prob_state_1'] = test_probs[:, 1]
    
    X_cols = [
        'hour', 'day_of_week', 'month', 'day_of_year', 
        'lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h',
        'prob_state_0', 'prob_state_1' # <-- Passing Advanced ML outputs into Neural Network!
    ]
    # Dynamically grab the target to support dataset swapping
    y_col = [col for col in train_df.columns if col not in X_cols][0]
    
    # Now continue entirely with Neural Network Logic
    print("Scaling and Sequencing Expanded Hybrid Matrix...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df[X_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[[y_col]])
    
    X_test_scaled = scaler_X.transform(test_df[X_cols])
    y_test_scaled = scaler_y.transform(test_df[[y_col]])
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    train_data = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
    test_data = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test_seq))
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    model = TimeSeriesTransformer(num_features=len(X_cols), d_model=64, nhead=4, num_layers=2, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print("Training Hybrid Transformer with Early Stopping limit = 20 epochs...")
    best_loss = float('inf')
    best_weights = None
    patience, patience_counter = 4, 0
    epochs = 20 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                val_loss += criterion(model(X_batch), y_batch).item()
                
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(test_loader)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping deployed at epoch {epoch+1}")
                break
                
    model.load_state_dict(best_weights)
    
    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            all_preds.append(model(X_batch).numpy())
            
    y_pred_scaled = np.vstack(all_preds)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test_seq).ravel()
    
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    
    print("\n" + "="*40)
    print("--- Model C (Hybrid Forecaster) Validation ---")
    print("="*40)
    print(f"Architecture : GMM Probabilities -> Transformer")
    print(f"Test Size    : {test_size} hours (3 months)")
    print(f"MAE          : {mae:.2f} MW")
    print(f"RMSE         : {rmse:.2f} MW")
    print("========================================")

if __name__ == "__main__":
    train_and_evaluate_hybrid("data/processed/featured_data.csv")
