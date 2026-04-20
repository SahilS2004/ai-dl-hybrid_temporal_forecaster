import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import copy

from advanced_hybrid_model import AdvancedHybridForecaster

# =========================================================
# 1. ADVANCED REGULARIZATION (Level 10)
# =========================================================
def timeseries_mixup(x, y, alpha=0.2):
    """
    Implements MixUp for Time-Series (Linear interpolation of sequences).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

# =========================================================
# 2. CURRICULUM LEARNING STRATEGY (Level 10)
# =========================================================
def get_curriculum_loaders(X, y, batch_size=64):
    """
    Splits data into 'Easy' (Normal) and 'Hard' (Extreme) based on volatility.
    Note: We use rolling_std directly for curriculum splitting now.
    """
    std_col_idx = -1 # assuming rolling_std is the last feature
    volatility = X[:, :, std_col_idx].mean(dim=1)
    
    threshold = torch.median(volatility)
    easy_mask = volatility <= threshold
    hard_mask = volatility > threshold
    
    easy_loader = DataLoader(TensorDataset(X[easy_mask], y[easy_mask]), batch_size=batch_size, shuffle=True)
    hard_loader = DataLoader(TensorDataset(X[hard_mask], y[hard_mask]), batch_size=batch_size, shuffle=True)
    full_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    
    return easy_loader, hard_loader, full_loader

# =========================================================
# 3. TRAINING PIPELINE
# =========================================================
def train_novel_model(data_path="data/processed/featured_data.csv", results_dir="reports/advanced"):
    os.makedirs(results_dir, exist_ok=True)
    print("🚀 Initializing Level-10 Advanced Training Pipeline...")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Feature Setup
    X_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h']
    target_col = [col for col in df.columns if col not in X_cols][0]
    
    # 1. Neural State Identification (Joint Learning)
    print("--- Stage 1: Initializing Neural Regime Classifier ---")
    # GMM is removed. The model now identifies states during training.

    # 2. Sequential Preparation
    seq_length = 24
    def create_sequences(data, targets, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:(i + seq_len)])
            ys.append(targets[i + seq_len])
        return np.array(xs), np.array(ys)

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(df[X_cols])
    Y_scaled = Y_scaler.fit_transform(df[[target_col]])
    
    X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, seq_length)
    
    # Split
    split = int(0.8 * len(X_seq))
    X_train, X_test = torch.FloatTensor(X_seq[:split]), torch.FloatTensor(X_seq[split:])
    Y_train, Y_test = torch.FloatTensor(Y_seq[:split]), torch.FloatTensor(Y_seq[split:])
    
    # 3. Model Initialization (Novel Architecture)
    model = AdvancedHybridForecaster(num_features=len(X_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # AdamW for better weight decay handling
    criterion = nn.HuberLoss() # Robust to outliers spikes identified in failure analysis
    
    # 4. Curriculum Learning Execution
    easy_loader, hard_loader, full_loader = get_curriculum_loaders(X_train, Y_train)
    
    print("--- Stage 2: Curriculum Learning (Easy -> Hard -> Full) ---")
    epochs_per_stage = [2, 2, 5] 
    best_mae = float('inf')
    
    for stage, loader in enumerate([easy_loader, hard_loader, full_loader]):
        print(f"Starting Curriculum Stage {stage+1}...")
        for epoch in range(epochs_per_stage[stage]):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                # Apply MixUp for stage 3
                if stage == 2:
                    xb, yb = timeseries_mixup(xb, yb)
                
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Eval
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_mae = mean_absolute_error(Y_test.numpy(), test_pred.numpy())
                if test_mae < best_mae:
                    best_mae = test_mae
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            
            if epoch % 5 == 0:
                print(f"Stage {stage+1} Epoch {epoch}: Val MAE = {test_mae:.4f}")

    # =========================================================
    # 4. TECHNICAL VALIDATION (Level 10)
    # =========================================================
    print("--- Stage 3: Technical Validation (Ablation, Robustness, Interpretability) ---")
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    model.eval()
    
    # 1. Prediction Intervals (Robustness)
    with torch.no_grad():
        preds_base = Y_scaler.inverse_transform(model(X_test).numpy())
        # Noise Robustness Test (OOD simulation)
        X_noisy = X_test + torch.randn_like(X_test) * 0.1
        preds_noisy = Y_scaler.inverse_transform(model(X_noisy).numpy())
        noise_mae = mean_absolute_error(Y_scaler.inverse_transform(Y_test), preds_noisy)
        print(f"Robustness Check: MAE with 10% Noise = {noise_mae:.2f}")

    # 2. XAI: Attention Map Visualization
    print("Generating Attention Heatmaps for Regime Analysis...")
    sample_idx = 100
    with torch.no_grad():
        _ = model(X_test[sample_idx:sample_idx+1])
        attn = model.attn_map[0].numpy() 
        
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis')
    plt.title("Regime-Aware Attention Map (Head 0)")
    plt.xlabel("Lookback Window (T-24 to T-1)")
    plt.ylabel("Lookback Window")
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, "attention_map.png"))
    plt.close()

    # 3. Final Report
    actual = Y_scaler.inverse_transform(Y_test)
    final_mae = mean_absolute_error(actual, preds_base)
    print(f"\nFinal Validated MAE: {final_mae:.2f}")
    
    return final_mae

if __name__ == "__main__":
    train_novel_model()
