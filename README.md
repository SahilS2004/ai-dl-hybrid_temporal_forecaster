# Hybrid Temporal Forecaster

A time-series forecasting pipeline combining Statistical/Probabilistic Advanced ML and sequence-based Deep Learning models to predict multi-step ahead values under changing data regimes.

## Project Structure
- `data/raw/`: Raw unedited datasets
- `data/processed/`: Normalized/cleaned dataset splits
- `notebooks/`: Exploratory Data Analysis (EDA) and ad-hoc scripts
- `src/data/`: Data loading, preprocessing, feature engineering
- `src/features/`: Complex regime features, lag extraction
- `src/models/`: Baseline stats, Adv. ML, DL and Hybrid integrations
- `src/utils/`: Common metrics (MAE, RMSE, F1), logging
- `reports/`: Latex/Markdown reports, architecture diagrams and Ablation tables

## Pipeline Focus
1. Baseline prediction
2. Advance ML Regression / Classification (HMMs, SVMs, ARIMA)
3. Deep Learning Architecture (Transformers, LSTM, CNN)
4. Hybrid Integration for State/Regime-Aware forecasting



Here is the exact cheat sheet of terminal commands you should use for your video recording.

To prove your project is dynamic and actually works, you should run these commands live (or record them running) one by one. I have broken down exactly what each command does and what accuracy/output it prints to the screen.

Command 1: The SVR Baseline (Model A)
The Command:

bash
venv/bin/python src/models/baseline_svr.py
What happens on screen:

It will print Loading data... and Scaling temporal and lag features....
It shows the classical Machine Learning algorithm mapping the data using the RBF Kernel.
The Accuracy Output: Within about 10 seconds, it will print a clean validation box showing the Test Size (3 months), the MAE (~869 MW), and the RMSE (~1094 MW).
Video Tip: Emphasize that this is your high-bias baseline to prove classical ML works, but has limits.
Command 2: The Deep Learning Baseline (Model B)
The Command:

bash
venv/bin/python src/models/baseline_transformer.py
What happens on screen:

It will initialize the PyTorch tensors and data loaders.
The Live Action: You will literally see the Neural Network training live! It will print Epoch 01 | Train Loss... | Val Loss... line by line.
It will also demonstrate Early Stopping. If the model stops improving, it will print Early Stopping deployed at epoch X.
The Accuracy Output: After training finishes, it evaluates the test set and prints the validation box showing an improved MAE (~861 MW).
Video Tip: Say, "Here we observe the self-attention mechanism slowly minimizing the loss function."
Command 3: The Hybrid Innovation (Model C)
The Command:

bash
venv/bin/python src/models/hybrid_gmm_transformer.py
What happens on screen:

First, it prints Training GMM (Probabilistic Model) to identify Extreme Regimes.... You can highlight that this is the clustering component assigning state probabilities!
Then, it concatenates those probabilities and prints Training Hybrid Transformer with Early Stopping limit = 20 epochs...
Again, you see the live Epoch training.
The Accuracy Output: It prints the final Model C validation block showing the final MAE and RMSE.
Video Tip: This is the climax of the demo. Point out how the loss starts much more stable because the Transformer now has "hints" from the GMM about whether the data is highly volatile string or a normal day.
Command 4: The Mathematical Failure Analysis
The Command:

bash
venv/bin/python src/models/failure_analysis.py
What happens on screen:

It calculates the absolute worst predictions your model made over the entire test set.
The Output: It prints a dramatic red banner: 🚨 TOP 5 MODEL FAILURE ZONES (Mathematical Analysis)
It lists the exact Dates, Shift Magnitudes (e.g., jumps of +8,189 MW in a single hour), and mathematical reasoning for why autoregressive lags mathematically break at that exact threshold.
Video Tip: This proves to the professors you are doing Master's level Deep Learning, not just running a library blindly.






