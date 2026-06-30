"""
M5 Forecasting - Training Script
- Works from ANY directory, no cd required
- Uses absolute paths relative to this script's location
- FAST_MODE=True for quick test (~5-10 min)
- FAST_MODE=False for full production training
"""
import os, sys, time, gc, warnings, joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Absolute paths (always works regardless of where you run from) ─────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../m5_forecast/model/
ROOT_DIR  = os.path.dirname(THIS_DIR)                    # .../m5_forecast/
DATA_DIR  = os.path.join(ROOT_DIR, "data", "raw")
OUT_DIR   = THIS_DIR   # model files saved next to this script

SALES_CSV    = os.path.join(DATA_DIR, "sales_train_validation.csv")
CALENDAR_CSV = os.path.join(DATA_DIR, "calendar.csv")
PRICES_CSV   = os.path.join(DATA_DIR, "sell_prices.csv")

# ── CONFIG ────────────────────────────────────────────────────────────────────
FAST_MODE     = True   # True = quick test (~5-10 min) | False = full training
MAX_ITEMS     = 500    # items sampled in FAST_MODE
LAST_DAYS     = 200    # days used in FAST_MODE
OPTUNA_TRIALS = 5      # Optuna trials in FAST_MODE  (15 for full)
LGB_ROUNDS    = 200    # LightGBM rounds in FAST_MODE (1000 for full)
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()

def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

def ram(df):
    return f"{df.memory_usage(deep=True).sum()/1e6:.0f} MB"

def check_data_files():
    missing = [f for f in [SALES_CSV, CALENDAR_CSV, PRICES_CSV] if not os.path.exists(f)]
    if missing:
        log("❌ ERROR: Missing data files:")
        for f in missing:
            log(f"   {f}")
        log(f"\n👉 Put your 3 CSV files inside: {DATA_DIR}")
        sys.exit(1)

log("=" * 65)
log("M5 Sales Forecasting — Training Script")
log(f"Mode     : {'FAST (subset)' if FAST_MODE else 'FULL'}")
log(f"Root dir : {ROOT_DIR}")
log(f"Data dir : {DATA_DIR}")
log(f"Output   : {OUT_DIR}")
log("=" * 65)

check_data_files()

# ── 1. Load sales ─────────────────────────────────────────────────────────────
log("Step 1/9: Reading column names...")
all_cols  = pd.read_csv(SALES_CSV, nrows=0).columns.tolist()
id_cols   = ["id","item_id","dept_id","cat_id","store_id","state_id"]
day_cols  = [c for c in all_cols if c.startswith("d_")]
n_days    = LAST_DAYS if FAST_MODE else 500
keep_days = day_cols[-n_days:]
use_cols  = id_cols + keep_days
log(f"  Total day columns: {len(day_cols)} | Using last: {len(keep_days)}")

log("Step 2/9: Loading sales CSV (30-60s)...")
sales = pd.read_csv(SALES_CSV, usecols=use_cols)
log(f"  Loaded: {sales.shape} | RAM: {ram(sales)}")

if FAST_MODE and len(sales) > MAX_ITEMS:
    log(f"  FAST_MODE: sampling {MAX_ITEMS} of {len(sales)} items...")
    sales = sales.sample(n=MAX_ITEMS, random_state=42).reset_index(drop=True)

log("Step 3/9: Loading calendar...")
calendar = pd.read_csv(CALENDAR_CSV, usecols=[
    "d","date","wm_yr_wk","weekday","wday","month",
    "event_name_1","event_type_1","event_name_2","event_type_2",
    "snap_CA","snap_TX","snap_WI"
])
log(f"  Calendar: {calendar.shape}")

log("  Loading sell_prices (194MB — filtering to used items only)...")
if FAST_MODE:
    needed = set(sales["item_id"].unique())
    chunks = []
    for chunk in pd.read_csv(PRICES_CSV, chunksize=500_000):
        f = chunk[chunk["item_id"].isin(needed)]
        if len(f): chunks.append(f)
    prices = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    log(f"  Prices filtered: {prices.shape}")
else:
    prices = pd.read_csv(PRICES_CSV)
    log(f"  Prices loaded: {prices.shape}")
gc.collect()

# ── 2. Melt ───────────────────────────────────────────────────────────────────
log(f"Step 4/9: Melting ({len(sales):,} items × {len(keep_days)} days)...")
df = sales.melt(id_vars=id_cols, var_name="d", value_name="sales")
del sales; gc.collect()
log(f"  Long format: {df.shape} | RAM: {ram(df)}")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
log("Step 5/9: Merging calendar...")
df = df.merge(calendar, on="d", how="left"); del calendar; gc.collect()
log("  Merging prices...")
df = df.merge(prices, on=["store_id","item_id","wm_yr_wk"], how="left")
del prices; gc.collect()
df.drop(columns=["id","d","wm_yr_wk"], errors="ignore", inplace=True)
log(f"  Merged: {df.shape} | RAM: {ram(df)}")

# ── 4. Cast dtypes ────────────────────────────────────────────────────────────
log("Step 6/9: Casting dtypes...")
df["date"]  = pd.to_datetime(df["date"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce").astype("float32")
df = df[df["sales"].notna()].copy()
for c in df.select_dtypes("float64").columns: df[c] = df[c].astype("float32")
for c in df.select_dtypes("int64").columns:   df[c] = df[c].astype("int16")
for c in df.select_dtypes("object").columns:  df[c] = df[c].astype("category")
df["dayofweek"]  = df["date"].dt.dayofweek.astype("int8")
df["month"]      = df["date"].dt.month.astype("int8")
df["year"]       = df["date"].dt.year.astype("int16")
df["is_weekend"] = df["dayofweek"].isin([5,6]).astype("int8")
gc.collect()
log(f"  Done | RAM: {ram(df)}")

# ── 5. Features ───────────────────────────────────────────────────────────────
log("Step 7/9: Building lag & rolling features...")
df.sort_values(["item_id","store_id","date"], inplace=True)
df.reset_index(drop=True, inplace=True)
grp = df.groupby(["item_id","store_id"])["sales"]

for lag in [1, 7, 14, 28]:
    log(f"  lag_{lag}...")
    df[f"lag_{lag}"] = grp.shift(lag).astype("float32")

log("  rolling mean 7...")
df["rmean_7"]  = grp.shift(1).rolling(7).mean().astype("float32").values
log("  rolling mean 28...")
df["rmean_28"] = grp.shift(1).rolling(28).mean().astype("float32").values

log("  price features...")
pg = df.groupby(["item_id","store_id"])["sell_price"]
df["price_mean"] = pg.transform("mean").astype("float32")
df["price_norm"] = ((df["sell_price"] - df["price_mean"]) /
                    pg.transform("std").fillna(1)).astype("float32")

log("  SNAP features...")
df["is_snap"] = np.int8(0)
for state, col in [("CA","snap_CA"),("TX","snap_TX"),("WI","snap_WI")]:
    if col in df.columns:
        mask = df["state_id"].astype(str) == state
        df.loc[mask, "is_snap"] = df.loc[mask, col].fillna(0).astype("int8")

gc.collect()
log(f"  Features built | RAM: {ram(df)}")

# ── 6. Encode cats ────────────────────────────────────────────────────────────
log("Step 8/9: Encoding categoricals...")
CAT_COLS = ["item_id","dept_id","cat_id","store_id","state_id",
            "weekday","event_name_1","event_type_1","event_name_2","event_type_2"]
cat_mappings = {}
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype("category")
        cat_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes.astype("int16")

joblib.dump(cat_mappings, os.path.join(OUT_DIR, "cat_mappings.pkl"))

df = df[df["lag_28"].notna()].copy()
df.reset_index(drop=True, inplace=True)
gc.collect()

split_date = df["date"].max() - pd.Timedelta(days=28)
train = df[df["date"] <  split_date].copy()
valid = df[df["date"] >= split_date].copy()
del df; gc.collect()

FEATURES = [c for c in train.columns if c not in ["sales","date"]]
TARGET   = "sales"
log(f"  Train: {len(train):,} | Valid: {len(valid):,} | Features: {len(FEATURES)}")

train_ds = lgb.Dataset(train[FEATURES], label=train[TARGET], free_raw_data=True)
valid_ds = lgb.Dataset(valid[FEATURES], label=valid[TARGET], free_raw_data=True, reference=train_ds)

# ── 7. Optuna + final model ───────────────────────────────────────────────────
n_trials = OPTUNA_TRIALS if FAST_MODE else 15
n_rounds = LGB_ROUNDS    if FAST_MODE else 1000
log(f"Step 9/9: Optuna tuning ({n_trials} trials, {n_rounds} max rounds)...")

def objective(trial):
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "verbosity":        -1,
        "num_threads":      4,
        "feature_pre_filter": False,
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq":     5,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 200),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-3, 3.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-3, 3.0, log=True),
    }
    m = lgb.train(params, train_ds, valid_sets=[valid_ds],
                  num_boost_round=n_rounds,
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)])
    rmse = np.sqrt(mean_squared_error(valid[TARGET], m.predict(valid[FEATURES])))
    log(f"  Trial {trial.number+1}/{n_trials}: RMSE={rmse:.4f}")
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)
log(f"  Best RMSE: {study.best_value:.5f}")

log("Training final model...")
best = {**study.best_params, "objective":"regression", "metric":"rmse",
        "verbosity":-1, "num_threads":4, "bagging_freq":5, "feature_pre_filter":False}
model = lgb.train(best, train_ds, valid_sets=[valid_ds],
                  num_boost_round=n_rounds,
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

rmse = np.sqrt(mean_squared_error(valid[TARGET], model.predict(valid[FEATURES])))
log(f"\n✅ Final RMSE: {rmse:.5f}")

joblib.dump(model,    os.path.join(OUT_DIR, "model.pkl"))
joblib.dump(FEATURES, os.path.join(OUT_DIR, "feature_list.pkl"))
log(f"✅ Saved to: {OUT_DIR}")
log(f"✅ Total time: {(time.time()-t0)/60:.1f} minutes")
