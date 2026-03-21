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
