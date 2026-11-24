from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "meta_asian_breakout.joblib"
DEFAULT_THRESHOLD = 0.45  # Adjust based on your Validation Set calibration

# Exact feature order used during LightGBM training
FEATURE_ORDER = [
    'direction', 'range_atr_ratio', 'breakout_dist_norm',
    'candle_body_ratio', 'candle_size_norm', 'wick_against_ratio',
    'dist_d_ema50_norm', 'vol_ratio', 'entry_hour', 'dayofweek'
]

# Global dictionary to hold the loaded model in RAM
ml_models = {}

# ==========================================
# LIFESPAN: LOAD MODEL INTO RAM ON STARTUP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained LightGBM model once when the server starts."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    
    ml_models["asian_breakout"] = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded {MODEL_PATH} into RAM successfully.")
    yield
    ml_models.clear()
    print("[INFO] Model cleared from RAM.")

app = FastAPI(title="MT4 Asian Breakout AI Bridge", lifespan=lifespan)

# ==========================================
# REQUEST & RESPONSE SCHEMAS
# ==========================================
class BreakoutFeatures(BaseModel):
    """Input schema matching the 10 breakout features from MT4."""
    direction: int = Field(..., description="1 for BUY, -1 for SELL")
    range_atr_ratio: float
    breakout_dist_norm: float
    candle_body_ratio: float
    candle_size_norm: float
    wick_against_ratio: float
    dist_d_ema50_norm: float
    vol_ratio: float
    entry_hour: int
    dayofweek: int
    threshold: float = Field(default=DEFAULT_THRESHOLD, description="Optional override from MT4 EA")

class PredictionResponse(BaseModel):
    """Simple JSON response easy to parse in MQL4."""
    allow_trade: int      # 1 = Allow, 0 = Reject (easier to parse in MQL4 than boolean)
    probability: float    # Win probability (0.0000 to 1.0000)
    threshold: float
    direction: int

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/health")
def health_check():
    """Verify server and model status."""
    return {
        "status": "ok",
        "model_loaded": "asian_breakout" in ml_models
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_signal(payload: BreakoutFeatures):
    """Evaluate breakout features and return trade authorization."""
    model = ml_models.get("asian_breakout")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Build single-row DataFrame preserving exact training column order
        input_data = {col: [getattr(payload, col)] for col in FEATURE_ORDER}
        X_df = pd.DataFrame(input_data)

        # Predict probability of Class 1 (WIN)
        prob_win = float(model.predict_proba(X_df)[0, 1])
        allow = 1 if prob_win >= payload.threshold else 0

        print(
            f"[PREDICT] Dir={payload.direction:+d} | Hour={payload.entry_hour} | "
            f"Prob={prob_win:.4f} | Thresh={payload.threshold:.2f} -> ALLOW={allow}"
        )

        return PredictionResponse(
            allow_trade=allow,
            probability=round(prob_win, 4),
            threshold=payload.threshold,
            direction=payload.direction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))