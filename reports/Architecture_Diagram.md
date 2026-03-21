# Hybrid Temporal Forecaster Architecture

This diagram visualizes the data flow required for **Model C (The Hybrid Forecaster)**, answering the rubric criteria: *"A high-level visual showing the data flow between the probabilistic model and the neural network."*

```mermaid
flowchart TD
    subgraph Data Pipeline
        Raw[Raw Energy Data] --> F[Feature Engineering]
        F --> Temporals[Temporal Features: Hour, Day]
        F --> Lags[Lag L_1, L_2, L_24]
        F --> Variances[Rolling Mean & Std Dev]
    end

    subgraph Advanced ML Module 
        Variances -->|Standardize| S[StandardScaler]
        S --> GMM((Gaussian Mixture Model))
        GMM -->|Predict State Probabilities| SP1[P_Normal_State]
        GMM -->|Predict State Probabilities| SP2[P_Extreme_State]
    end

    subgraph Hybrid Deep Learning Fusion
        Temporals --> Concat[Feature Matrix Concatenation]
        Lags --> Concat
        Variances --> Concat
        SP1 --> Concat
        SP2 --> Concat
        
        Concat --> Seq[Create 24-hr Sequence Window]
    end

    subgraph Time-Series Deep Learning
        Seq --> Proj[Linear Projection Layer]
        Proj --> PE[Sine/Cosine Positional Encoding]
        
        PE --> TBlock1[Transformer Block: Self-Attention]
        TBlock1 --> TBlock2[Transformer Block: Feed Forward & Dropout]
        
        TBlock2 --> Out[Linear Output Layer]
    end

    Out --> Final[Final MW_Load Forecast T+1]

    %% Styles
    classDef prob fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef nn fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef fused fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    
    class GMM,SP1,SP2 prob;
    class TBlock1,TBlock2,PE nz;
    class Out,Final nn;
    class Concat fused;
```

### Explanation of the Hybrid Data Flow
1. **Classical Feature Extraction:** Standard rolling constraints map the temporal dependencies.
2. **Probabilistic Interpretation:** A `Gaussian Mixture Model` isolates exactly what "regime" the dataset is currently experiencing (e.g. Normal Grid Operation vs Summer Heatwave). It outputs explicit probabilities, entirely decoding the "black box" nature of standard Deep Learning distributions.
3. **Neural Representation Learning:** The Transformer encoder receives these probabilistic mappings as external "hints," alongside the raw positional embeddings. Its *Multi-Head Self-Attention Mechanism* then learns to shift its internal interpolation weights (e.g., relying strictly on `lag_24` during normal states, but focusing exclusively on `lag_1` during extreme state heatwaves).
