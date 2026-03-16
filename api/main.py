from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager 
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
import logging
from api.schema import OrderInput, PredictionOutput, BatchPredictionOutput

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

    # Load directly from pkl — works everywhere!
    import xgboost as xgb
    model        = joblib.load("models/best_model.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

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
@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(orders: list[OrderInput]):
    try:
        start_time   = time.time()
        predictions  = []
        high_risk    = 0

        for order in orders:
            input_df    = prepare_input(order)
            probability = float(model.predict_proba(input_df)[0][1])
            prediction  = int(probability >= 0.5)
            risk_level  = get_risk_level(probability)

            if risk_level == "High 🔴":
                high_risk += 1

            predictions.append(PredictionOutput(
                is_return          = prediction,
                return_probability = round(probability, 4),
                risk_level         = risk_level
            ))

        response_time = (time.time() - start_time) * 1000
        logger.info(f"Batch: {len(orders)} orders | "
                   f"High risk: {high_risk} | "
                   f"Time: {response_time:.2f}ms")

        return BatchPredictionOutput(
            predictions      = predictions,
            total_orders     = len(orders),
            high_risk_count  = high_risk
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ─────────────────────────────────────────
# ENDPOINT 4 — Model Info
# ─────────────────────────────────────────
@app.get("/model/info")
def model_info():
    return {
        "model_name"    : "ReturnPredictor",
        "model_type"    : "XGBoost Classifier",
        "version"       : "1.0.0",
        "features"      : len(feature_cols),
        "feature_names" : feature_cols,
        "metrics"       : {
            "pr_auc"    : 0.8134,
            "recall"    : 0.6911,
            "f1_score"  : 0.8053,
            "precision" : 0.9647
        },
        "description": "Predicts e-commerce order returns before shipping"
    }