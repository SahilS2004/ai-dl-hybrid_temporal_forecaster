import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class RegimeGatedAttention(nn.Module):
    """
    Novel Architectural Block (Level 10): 
    Regime-Gated Attention modulates standard self-attention weights 
    based on exogenous regime probabilities (GMM outputs).
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(RegimeGatedAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.regime_gate = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, regime_probs):
        # regime_probs shape: [Batch, 2] -> We expand it to match sequence length
        # Or we use the regime_probs at each timestep if they were fed as features.
        # Here we assume regime_probs is [Seq, Batch, 2]
        
        # 1. Standard Attention
        attn_output, attn_weights = self.attn(x, x, x)
        
        # 2. Gating mechanism
        # regime_probs is [Seq, Batch, 2]. We project it to d_model.
        gate = self.regime_gate(regime_probs) # [Seq, Batch, d_model]
        
        # 3. Modulate the output
        gated_output = attn_output * gate
        
        return self.norm(x + self.dropout(gated_output)), attn_weights

class AdvancedHybridForecaster(nn.Module):
    """
    State-of-the-Art Temporal Forecaster.
    Incorporates:
    - Multi-head Regime-Gated Attention
    - Skip Connections & LayerNorm
    - Multi-Scale Convolutional Feature Extraction
    - Support for Explainability (Attn Weights collection)
    """
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(AdvancedHybridForecaster, self).__init__()
        
        # 1. Multi-Scale Embedding (Captures local patterns)
        self.conv1 = nn.Conv1d(num_features, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_features, d_model, kernel_size=5, padding=2)
        self.input_projection = nn.Linear(d_model * 2, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Custom Regime-Gated Attention Layers
        self.layers = nn.ModuleList([
            RegimeGatedAttention(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        
        # 3. Output Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.attn_map = None # For XAI
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p)

    def forward(self, x, regime_probs):
        """
        x: [Batch, Seq_Len, Features]
        regime_probs: [Batch, Seq_Len, 2] (Extracted from GMM)
        """
        # Conv layers expect [Batch, Features, Seq_Len]
        x_in = x.transpose(1, 2)
        c1 = F.relu(self.conv1(x_in))
        c2 = F.relu(self.conv2(x_in))
        
        # Concatenate and project back to d_model
        x = torch.cat([c1, c2], dim=1).transpose(1, 2) # [Batch, Seq, 2*d_model]
        x = self.input_projection(x) # [Batch, Seq, d_model]
        
        # Sequence first for Transformer compatibility [Seq, Batch, d_model]
        x = x.transpose(0, 1)
        regime_probs = regime_probs.transpose(0, 1) # [Seq, Batch, 2]
        
        x = self.pos_encoder(x)
        
        # Iterate through custom layers
        for layer in self.layers:
            x, attn_weights = layer(x, regime_probs)
            self.attn_map = attn_weights # Store last layer attention for XAI
            
        # Global selection: Take the last timestep
        out = x[-1, :, :]
        return self.fc(out)

if __name__ == "__main__":
    # Test forward pass
    batch_size = 16
    seq_len = 24
    num_f = 10
    model = AdvancedHybridForecaster(num_features=num_f)
    sample_x = torch.randn(batch_size, seq_len, num_f)
    sample_regime = torch.randn(batch_size, seq_len, 2)
    output = model(sample_x, sample_regime)
    print(f"Output shape: {output.shape}") # Should be [16, 1]
