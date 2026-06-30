#!/bin/bash
echo "============================================"
echo " M5 Sales Forecast - Setup and Run"
echo "============================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt || { echo "pip install failed"; exit 1; }

echo ""
echo "[2/3] Training the model (5-10 min in FAST_MODE)..."
python model/train_optimized.py || { echo "Training failed"; exit 1; }

echo ""
echo "[3/3] Starting API at http://localhost:8000"
echo "      Open frontend/index.html in your browser!"
echo "      Press Ctrl+C to stop."
python -m uvicorn backend.main:app --reload --port 8000 --app-dir .
