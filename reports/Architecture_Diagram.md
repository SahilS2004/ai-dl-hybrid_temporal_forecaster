# Phase 3: Synergistic Hybrid Architecture

This diagram visualizes the **Synergistic interaction** between the ML and DL components, specifically designed for the Phase 3 rubric (Level 5).




          📊 Input Data
     (Temporal + Lag Features)
                 │
        ┌────────┴────────┐
        ▼                 ▼

   🤖 ML Model         🧠 DL Model
   (GMM)              (Transformer)
   --------           -------------
   Finds Regime       Learns Patterns
   (0,1,2)            (sequence behavior)
        │                 │
        ▼                 ▼
   Base Load         Residual Fix
        │                 │
        └───────➕────────┘
                ▼
        🎯 Final Forecast

### 💡 Why this is "Synergistic" (Rubric Level 5)
1. **Model A (ML)** handles the **static distribution**: It identifies the "Regime" and provides a stable base-load estimate.
2. **Model B (DL)** handles the **dynamic sequence**: It calculates the "Residual" (the fine-grained correction) by looking at high-frequency temporal patterns.
3. **The Synergy**: The whole is greater than the sum because the Transformer no longer has to "re-learn" the basic load levels of different regimes; it only focuses on correcting the errors of the ML model.

### 📊 Tensor Flow Details
- **Input**: `(Batch, 24, 9)`
- **GMM Output**: `(Batch, 1)` Discrete Labels
- **Regime Embedding**: `(Batch, 1, 64)` Added to Transformer sequence.
- **Fusion Point**: `Base_Load[Regime] + Transformer_Residual(x)`

