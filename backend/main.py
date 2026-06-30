"""
M5 Forecast API - FastAPI Backend
Endpoints:
  GET  /              → health check
  POST /predict       → single-row forecast
  POST /predict/batch → multi-row forecast
  GET  /features      → list of expected features
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pandas as pd
import joblib
import os

# ── Absolute paths (works from any directory) ─────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(THIS_DIR)
MODEL_DIR  = os.path.join(ROOT_DIR, "model")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="M5 Walmart Forecast API",
    description="LightGBM-powered sales forecasting for M5 dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH",    os.path.join(MODEL_DIR, "model.pkl"))
FEATURES_PATH    = os.getenv("FEATURES_PATH", os.path.join(MODEL_DIR, "feature_list.pkl"))
CAT_MAPPINGS_PATH = os.path.join(MODEL_DIR, "cat_mappings.pkl")

model        = None
FEATURES     = None
CAT_MAPPINGS = {}   # col -> {code: label}  (we invert to label->code for encoding)
CAT_ENCODERS = {}   # col -> {label: code}

@app.on_event("startup")
def load_model():
    global model, FEATURES, CAT_MAPPINGS, CAT_ENCODERS
    try:
        model    = joblib.load(MODEL_PATH)
        FEATURES = joblib.load(FEATURES_PATH)
        print(f"✅ Model loaded | Features: {len(FEATURES)}")
    except FileNotFoundError as e:
        print(f"⚠️  Model file not found: {e}. Run train_optimized.py first.")

    try:
        CAT_MAPPINGS = joblib.load(CAT_MAPPINGS_PATH)
        # Invert: code->label  becomes  label->code
        CAT_ENCODERS = {col: {v: k for k, v in mapping.items()}
                        for col, mapping in CAT_MAPPINGS.items()}
        print(f"✅ Cat mappings loaded for: {list(CAT_ENCODERS.keys())}")
    except FileNotFoundError:
        print("⚠️  cat_mappings.pkl not found — unknown categories will use -1")


def encode_cat(col: str, value: str) -> int:
    """Encode a categorical value using training mappings.
    Unknown values get -1 (LightGBM treats this as 'other/unseen')."""
    if col not in CAT_ENCODERS:
        return -1
    return CAT_ENCODERS[col].get(str(value), -1)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    item_id:   str  = Field(..., example="FOODS_3_090")
    store_id:  str  = Field(..., example="CA_1")
    dept_id:   str  = Field(..., example="FOODS_3")
    cat_id:    str  = Field(..., example="FOODS")
    state_id:  str  = Field(..., example="CA")

    date: str = Field(..., example="2016-05-22")

    lag_1:  float = Field(..., example=2.0)
    lag_7:  float = Field(..., example=3.0)
    lag_14: float = Field(..., example=2.5)
    lag_21: float = Field(0.0, example=2.0)
    lag_28: float = Field(..., example=3.5)
    lag_35: float = Field(0.0, example=2.0)
    lag_42: float = Field(0.0, example=1.5)

    rmean_7:  float = Field(..., example=2.3)
    rmean_14: float = Field(0.0, example=2.5)
    rmean_28: float = Field(..., example=2.4)
    rstd_7:   float = Field(0.0, example=0.5)
    rstd_14:  float = Field(0.0, example=0.6)
    rstd_28:  float = Field(0.0, example=0.7)

    rolling_mean_7:  float = Field(..., example=2.3)
    rolling_mean_28: float = Field(..., example=2.4)

    momentum_7:    Optional[float] = Field(None, example=1.1)
    momentum_14:   Optional[float] = Field(None, example=0.9)
    momentum_roll: Optional[float] = Field(None, example=1.0)

    sell_price:   Optional[float] = Field(None,  example=1.98)
    price_lag_1:  Optional[float] = Field(None,  example=1.98)
    price_change: Optional[float] = Field(1.0,   example=1.0)
    price_mean:   Optional[float] = Field(None,  example=2.0)
    price_std:    Optional[float] = Field(None,  example=0.1)
    price_norm:   Optional[float] = Field(0.0,   example=0.0)

    event_name_1:  Optional[str] = Field(None, example=None)
    event_type_1:  Optional[str] = Field(None, example=None)
    event_name_2:  Optional[str] = Field(None, example=None)
    event_type_2:  Optional[str] = Field(None, example=None)

    snap_CA: Optional[int] = Field(0, example=1)
    snap_TX: Optional[int] = Field(0, example=0)
    snap_WI: Optional[int] = Field(0, example=0)
    is_snap: Optional[int] = Field(0, example=1)

    weekday: Optional[str] = Field(None, example="Sunday")


class PredictResponse(BaseModel):
    item_id:    str
    store_id:   str
    date:       str
    prediction: float
    message:    str = "success"


class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[float]
    count:       int


# ── Helper ────────────────────────────────────────────────────────────────────
def build_row(req: PredictRequest) -> pd.DataFrame:
    """Turn a PredictRequest into a one-row DataFrame ready for LightGBM.
    All categoricals are integer-encoded using training mappings.
    Unknown values are encoded as -1 (safe fallback for LightGBM)."""
    date = pd.to_datetime(req.date)

    # Resolve event strings — treat None/empty/"NoEvent" all as the
    # same unknown bucket so encode_cat returns -1 cleanly
    def clean_event(v):
        if not v or v.strip().lower() in ("", "noevent", "none", "no event"):
            return "__unknown__"
        return v.strip()

    row = {
        # Categoricals — encoded as int using training mappings
        "item_id":      encode_cat("item_id",      req.item_id),
        "dept_id":      encode_cat("dept_id",      req.dept_id),
        "cat_id":       encode_cat("cat_id",       req.cat_id),
        "store_id":     encode_cat("store_id",     req.store_id),
        "state_id":     encode_cat("state_id",     req.state_id),
        "weekday":      encode_cat("weekday",      req.weekday or date.strftime("%A")),
        "event_name_1": encode_cat("event_name_1", clean_event(req.event_name_1)),
        "event_type_1": encode_cat("event_type_1", clean_event(req.event_type_1)),
        "event_name_2": encode_cat("event_name_2", clean_event(req.event_name_2)),
        "event_type_2": encode_cat("event_type_2", clean_event(req.event_type_2)),

        # Calendar numerics
        "dayofweek":  date.dayofweek,
        "month":      date.month,
        "year":       date.year,
        "is_weekend": int(date.dayofweek in [5, 6]),

        # Lags
        "lag_1":  req.lag_1,
        "lag_7":  req.lag_7,
        "lag_14": req.lag_14,
        "lag_28": req.lag_28,

        # Rolling
        "rmean_7":  req.rmean_7,
        "rmean_28": req.rmean_28,

        # Price
        "sell_price": req.sell_price if req.sell_price is not None else np.nan,
        "price_mean": req.price_mean if req.price_mean is not None else np.nan,
        "price_norm": req.price_norm if req.price_norm is not None else 0.0,

        # SNAP
        "is_snap": req.is_snap or 0,
    }

    df = pd.DataFrame([row])

    # Align exactly to trained feature list — add missing cols as NaN, drop extras
    if FEATURES:
        for col in FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        df = df[FEATURES]

    # All columns should be numeric now (no pandas category dtype)
    return df


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "n_features":   len(FEATURES) if FEATURES else 0,
    }


@app.get("/features", tags=["Info"])
def get_features():
    if not FEATURES:
        raise HTTPException(503, "Model not loaded yet")
    return {"features": FEATURES, "count": len(FEATURES)}


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run train_optimized.py first.")
    try:
        row  = build_row(req)
        pred = float(model.predict(row)[0])
        pred = max(0.0, round(pred, 4))
        return PredictResponse(
            item_id=req.item_id,
            store_id=req.store_id,
            date=req.date,
            prediction=pred,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predict"])
def predict_batch(req: BatchPredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")
    if not req.rows:
        raise HTTPException(400, "No rows provided.")
    try:
        frames = [build_row(r) for r in req.rows]
        df     = pd.concat(frames, ignore_index=True)
        preds  = model.predict(df)
        preds  = [max(0.0, round(float(p), 4)) for p in preds]
        return BatchPredictResponse(predictions=preds, count=len(preds))
    except Exception as e:
        raise HTTPException(500, f"Batch prediction error: {str(e)}")
