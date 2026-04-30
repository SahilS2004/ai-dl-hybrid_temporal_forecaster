import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMRegimeDetector:
    """
    ML Component (Gaussian Mixture Model):
    Learns to categorize the energy grid into states (e.g., Normal, Peak, Spike).
    """
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.regime_means = None

    def fit(self, X_variance):
        # We fit on variance/load to find regimes
        self.gmm.fit(X_variance)
        self.regime_means = self.gmm.means_
        return self

    def predict_probs(self, X_variance):
        return self.gmm.predict_proba(X_variance)

class ResidualTransformer(nn.Module):
    """
    DL Component (Transformer):
    Learns to predict the 'Residual' (the error) of the base regime prediction.
    """
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2):
        super(ResidualTransformer, self).__init__()
        # Input: [Batch, Seq, Features]
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Regime Embedding: Learns a unique vector for each of the 3 GMM regimes
        self.regime_embedding = nn.Embedding(3, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x, regime_labels):
        # x: [Batch, Seq, Features]
        # regime_labels: [Batch] -> the detected regime for the current window
        
        x = self.input_projection(x) # [Batch, Seq, d_model]
        
        # Inject Regime Context
        r_emb = self.regime_embedding(regime_labels).unsqueeze(1) # [Batch, 1, d_model]
        x = x + r_emb # Broadcast regime context across all timesteps
        
        x = self.transformer(x)
        
        # Take the last timestep and predict the adjustment
        return self.output_head(x[:, -1, :])

class SynergisticForecaster(nn.Module):
    """
    The Full Phase 3 Hybrid: 
    Synergy = Base Regime Load (ML) + Temporal Correction (DL)
    """
    def __init__(self, num_features):
        super(SynergisticForecaster, self).__init__()
        self.transformer = ResidualTransformer(num_features)
        
        # Base loads for each regime (can be learned or static)
        self.base_loads = nn.Parameter(torch.zeros(3)) 

    def forward(self, x, regime_labels):
        # 1. ML Contribution: Get the base load for this regime
        base_load = self.base_loads[regime_labels].unsqueeze(1)
        
        # 2. DL Contribution: Get the fine-tuned adjustment
        adjustment = self.transformer(x, regime_labels)
        
        # 3. Synergy: Combine them
        return base_load + adjustment

if __name__ == "__main__":
    print("Testing Phase 3 Synergistic Model...")
    model = SynergisticForecaster(num_features=9)
    dummy_x = torch.randn(8, 24, 9)
    dummy_regimes = torch.randint(0, 3, (8,))
    output = model(dummy_x, dummy_regimes)
    print(f"Output Shape: {output.shape}") # [8, 1]
    print("Logic: BaseLoad(Regime) + TransformerCorrection = Forecast")
