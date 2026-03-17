from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager 
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
import logging
from api.schema import OrderInput, PredictionOutput, BatchPredictionOutput
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Paths from environment
MODEL_PATH        = os.getenv("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH       = os.getenv("SCALER_PATH", "models/scaler.pkl")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "models/feature_cols.pkl")
METRICS_PATH      = os.getenv("METRICS_PATH", "models/metrics.json")
# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# GLOBAL MODEL STORAGE
# ─────────────────────────────────────────
model = None
scaler = None
feature_cols = None

# ─────────────────────────────────────────
# STARTUP — Load model when API starts
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_cols

    logger.info("Loading model from pkl files...")

    model        = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    logger.info("✅ Model loaded successfully!")
    logger.info(f"✅ Features: {len(feature_cols)}")

    yield

    logger.info("Shutting down API...")

# ─────────────────────────────────────────
# CREATE FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="E-Commerce Return Predictor",
    description="Predicts whether an order will be returned before shipping!",
    version="1.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low 🟢"
    elif probability < 0.6:
        return "Medium 🟡"
    else:
        return "High 🔴"



def prepare_input(order: OrderInput) -> pd.DataFrame:
    data = pd.DataFrame([order.model_dump()])
    data = data[feature_cols]
    return data

#End point -1
@app.get("/")
def health_check():
    return {
        "status"  : "healthy ✅",
        "model"   : "ReturnPredictor",
        "version" : "1.0.0",
        "message" : "E-Commerce Return Prediction API is running!"
    }

#End point - 2
@app.post("/predict", response_model=PredictionOutput)
def predict(order: OrderInput):
    try:
        start_time = time.time()

        # Prepare input
        input_df = prepare_input(order)

        # Predict
        probability = float(model.predict_proba(input_df)[0][1])
        prediction  = int(probability >= 0.5)
        risk_level  = get_risk_level(probability)

        # Log response time
        response_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction: {prediction} | "
                   f"Probability: {probability:.4f} | "
                   f"Risk: {risk_level} | "
                   f"Time: {response_time:.2f}ms")

        return PredictionOutput(
            is_return          = prediction,
            return_probability = round(probability, 4),
            risk_level         = risk_level
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#end point 3
# ─────────────────────────────────────────
# ENDPOINT 3 — Batch Prediction (Vectorized)
# ─────────────────────────────────────────
@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(orders: list[OrderInput]):
    try:
        start_time = time.time()

        # ── Vectorized — process ALL orders at once! ──
        # Build one DataFrame from all orders
        input_df = pd.DataFrame([order.model_dump() for order in orders])
        input_df = input_df[feature_cols]

        # Predict ALL at once — much faster!
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions   = (probabilities >= 0.5).astype(int)

        # Build results
        results    = []
        high_risk  = 0

        for prob, pred in zip(probabilities, predictions):
            risk_level = get_risk_level(float(prob))
            if risk_level == "High 🔴":
                high_risk += 1
            results.append(PredictionOutput(
                is_return          = int(pred),
                return_probability = round(float(prob), 4),
                risk_level         = risk_level
            ))

        response_time = (time.time() - start_time) * 1000
        logger.info(f"Batch: {len(orders)} orders | "
                   f"High risk: {high_risk} | "
                   f"Time: {response_time:.2f}ms")

        return BatchPredictionOutput(
            predictions     = results,
            total_orders    = len(orders),
            high_risk_count = high_risk
        )

    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# ─────────────────────────────────────────
# ENDPOINT 4 — Model Info
# ─────────────────────────────────────────
@app.get("/model/info")
def model_info():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {"pr_auc": 0.8134}

    return {
        "model_name"    : "ReturnPredictor",
        "model_type"    : "XGBoost Classifier",
        "version"       : "1.0.0",
        "features"      : len(feature_cols),
        "feature_names" : feature_cols,
        "metrics"       : metrics,
        "description"   : "Predicts e-commerce order returns before shipping"
    }