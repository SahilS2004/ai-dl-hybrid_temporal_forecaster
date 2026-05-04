import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture

# -------------------------------
# 1. GMM Regime Detector (Improved)
# -------------------------------
class GMMRegimeDetector:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)

    def fit(self, X):
        self.gmm.fit(X)
        return self

    def predict(self, X):
        return self.gmm.predict(X)

    def predict_probs(self, X):
        return self.gmm.predict_proba(X)


# -------------------------------
# 2. Positional Encoding (CRITICAL)
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------------
# 3. Residual Transformer (Improved)
# -------------------------------
class ResidualTransformer(nn.Module):
    def __init__(self, num_features, n_regimes, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.regime_embedding = nn.Embedding(n_regimes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Better pooling instead of last timestep
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(d_model, 1)

    def forward(self, x, regime_probs):
        """
        x: [B, T, F]
        regime_probs: [B, n_regimes]
        """

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Soft regime embedding (weighted sum)
        regime_ids = torch.arange(regime_probs.shape[1], device=x.device)
        regime_embs = self.regime_embedding(regime_ids)  # [R, d_model]

        weighted_emb = regime_probs @ regime_embs  # [B, d_model]
        weighted_emb = weighted_emb.unsqueeze(1)

        x = x + weighted_emb

        x = self.transformer(x)

        # Pool across sequence
        x = x.permute(0, 2, 1)  # [B, d_model, T]
        x = self.pool(x).squeeze(-1)

        return self.head(x)


# -------------------------------
# 4. Final Hybrid Model (PRO)
# -------------------------------
class SynergisticForecaster(nn.Module):
    def __init__(self, num_features, n_regimes=3):
        super().__init__()

        self.n_regimes = n_regimes

        self.transformer = ResidualTransformer(
            num_features=num_features,
            n_regimes=n_regimes
        )

        # Base loads initialized smarter
        self.base_loads = nn.Parameter(torch.randn(n_regimes))

    def forward(self, x, regime_probs):
        """
        regime_probs: [B, R]
        """

        # Soft base load (weighted)
        base = (regime_probs * self.base_loads).sum(dim=1, keepdim=True)

        correction = self.transformer(x, regime_probs)

        return base + correction



