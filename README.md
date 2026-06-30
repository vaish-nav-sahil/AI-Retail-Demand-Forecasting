# AI-Retail-Demand-Forecasting
This project focuses on building a memory-efficient and high-performance demand forecasting system using the M5 Forecasting dataset and LightGBM.  The goal is to predict future retail sales while handling:  Large-scale time series data Memory constraints (16GB RAM) Real-world retail patterns (seasonality, trends, price impact)




# M5 Walmart Sales Forecast

LightGBM-powered sales prediction with FastAPI backend + HTML frontend.

---

## Quick Start (3 steps)

### Step 1 — Put your data files here:

```
data/raw/sales_train_validation.csv   (115 MB)
data/raw/calendar.csv                 (102 KB)
data/raw/sell_prices.csv              (194 MB)
```

### Step 2 — Run everything with one command

**Windows** — double-click `run.bat`, OR in any terminal:

```
run.bat
```

**Mac / Linux:**

```bash
bash run.sh
```

That's it. The script installs packages, trains the model, and starts the API.

---

## Manual Steps (if you prefer)

Open a terminal inside this folder, then:

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Train the model (run from THIS folder, not a subfolder)
python model/train_optimized.py

# 3. Start the API (run from THIS folder)
python -m uvicorn backend.main:app --reload --port 8000 --app-dir .
```

Then open `frontend/index.html` in your browser.

---

## Config (model/train_optimized.py)

| Setting         | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `FAST_MODE`     | `True`  | `True` = 5-10 min test. `False` = full training (hours) |
| `MAX_ITEMS`     | `500`   | Items sampled in FAST_MODE                              |
| `LAST_DAYS`     | `200`   | Days used in FAST_MODE                                  |
| `OPTUNA_TRIALS` | `5`     | Hyperparameter search trials                            |
| `LGB_ROUNDS`    | `200`   | Max LightGBM boosting rounds                            |

Change `FAST_MODE = False` when you're ready for production accuracy.

---

## Project Structure

```
m5_forecast/
├── data/raw/             ← put your 3 CSV files here
├── model/
│   └── train_optimized.py
├── backend/
│   └── main.py           ← FastAPI server
├── frontend/
│   └── index.html        ← open in browser after training
├── requirements.txt
├── run.bat               ← Windows one-click run
└── run.sh                ← Mac/Linux one-click run
```

After training, these files appear in `model/`:

- `model.pkl`
- `feature_list.pkl`
- `cat_mappings.pkl`

---

## API

| Method | Endpoint         | Description        |
| ------ | ---------------- | ------------------ |
| GET    | `/`              | Health check       |
| GET    | `/features`      | List feature names |
| POST   | `/predict`       | Single prediction  |
| POST   | `/predict/batch` | Batch predictions  |

Docs at: http://localhost:8000/docs


Dataset is available on Kaggle
