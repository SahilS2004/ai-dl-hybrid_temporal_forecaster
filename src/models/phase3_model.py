# synergistic_forecaster.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# =====================================================
# 1. GMM Regime Detector
# =====================================================
class GMMRegimeDetector:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)

    def fit(self, X):
        self.gmm.fit(X)
        return self

    def predict_probs(self, X):
        return self.gmm.predict_proba(X)


# =====================================================
# 2. Sequence Creator
# =====================================================
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    return np.array(X_seq), np.array(y_seq)


# =====================================================
# 3. Positional Encoding (FIXED)
# =====================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =====================================================
# 4. Transformer Model
# =====================================================
class ResidualTransformer(nn.Module):
    def __init__(self, num_features, n_regimes, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.regime_embedding = nn.Embedding(n_regimes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, regime_probs):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        regime_ids = torch.arange(regime_probs.shape[1], device=x.device)
        regime_embs = self.regime_embedding(regime_ids)

        weighted_emb = regime_probs @ regime_embs
        weighted_emb = weighted_emb.unsqueeze(1)

        x = x + weighted_emb

        x = self.transformer(x)

        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)

        return self.head(x)


# =====================================================
# 5. Final Hybrid Model
# =====================================================
class SynergisticForecaster(nn.Module):
    def __init__(self, num_features, n_regimes=3):
        super().__init__()

        self.transformer = ResidualTransformer(num_features, n_regimes)
        self.base_loads = nn.Parameter(torch.zeros(n_regimes))

    def forward(self, x, regime_probs):
        base = (regime_probs * self.base_loads).sum(dim=1, keepdim=True)
        correction = self.transformer(x, regime_probs)
        return base + correction


# =====================================================
# 6. Training & Evaluation
# =====================================================
def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for X, y, r in loader:
            X, r = X.to(device), r.to(device)
            out = model(X, r)

            preds.extend(out.cpu().numpy())
            actuals.extend(y.numpy())

    return mean_squared_error(actuals, preds)


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y, r in train_loader:
            X, y, r = X.to(device), y.to(device), r.to(device)

            optimizer.zero_grad()
            preds = model(X, r)

            loss = loss_fn(preds.squeeze(), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


# =====================================================
# 7. MAIN (RUN PIPELINE)
# =====================================================
if __name__ == "__main__":

    # ---- Synthetic Data (replace with your real dataset) ----
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = X.sum(axis=1) + np.random.randn(1000)

    # ---- Scaling ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- GMM ----
    gmm = GMMRegimeDetector(n_components=3)
    gmm.fit(X_scaled)
    regime_probs = gmm.predict_probs(X_scaled)

    # ---- Sequence Creation ----
    seq_len = 20
    X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
    r_seq = regime_probs[seq_len:]

    # ---- Train-Test Split ----
    split = int(0.8 * len(X_seq))

    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]
    r_train, r_val = r_seq[:split], r_seq[split:]

    # ---- DataLoaders ----
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train),
                      torch.FloatTensor(y_train),
                      torch.FloatTensor(r_train)),
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val),
                      torch.FloatTensor(y_val),
                      torch.FloatTensor(r_val)),
        batch_size=32
    )

    # ---- Model ----
    model = SynergisticForecaster(num_features=X.shape[1], n_regimes=3)

    # ---- Train ----
    train_model(model, train_loader, val_loader)


