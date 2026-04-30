#!/bin/bash

echo "⚡ Starting Phase 3: Synergistic Hybrid Forecaster Pipeline ⚡"

# 1. Setup Environment
echo "📦 Installing requirements..."
./venv/bin/pip install -r requirements.txt

# 2. Feature Engineering
echo "🔍 Engineering Features..."
./venv/bin/python src/features/build_features.py

# 3. Training
echo "🧠 Training Synergistic Model..."
./venv/bin/python src/models/train_phase3.py

# 4. Ablation Study
echo "🧪 Running Diagnostic Ablation Study..."
./venv/bin/python src/models/run_phase3_ablation.py

