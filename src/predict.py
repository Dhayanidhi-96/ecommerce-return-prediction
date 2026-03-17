import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_model(model_path: str = None):
    """Load XGBoost model from pkl file"""
    path = model_path or os.getenv("MODEL_PATH", "models/best_model.pkl")
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model


def load_scaler(scaler_path: str = None):
    """Load StandardScaler from pkl file"""
    path = scaler_path or os.getenv("SCALER_PATH", "models/scaler.pkl")
    scaler = joblib.load(path)
    print(f"✅ Scaler loaded from {path}")
    return scaler


def load_feature_cols(feature_path: str = None):
    """Load feature column names"""
    path = feature_path or os.getenv(
        "FEATURE_COLS_PATH", "models/feature_cols.pkl")
    feature_cols = joblib.load(path)
    print(f"✅ Feature cols loaded — {len(feature_cols)} features")
    return feature_cols


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def preprocess_input(
    data: dict,
    feature_cols: list
) -> pd.DataFrame:
    """
    Preprocess a single order input dict into model-ready DataFrame
    Drops leakage columns and aligns with training features
    """
    leakage_cols = [
        'total_delivery_days',
        'delivery_delay_days',
        'is_late_delivery',
        'days_to_ship'
    ]

    df = pd.DataFrame([data])

    # Drop leakage if present
    df = df.drop(columns=[
        c for c in leakage_cols if c in df.columns
    ])

    # Align with training features
    df = df[feature_cols]

    return df


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low 🟢"
    elif probability < 0.6:
        return "Medium 🟡"
    else:
        return "High 🔴"


def predict_single(
    model,
    data: dict,
    feature_cols: list
) -> dict:
    """
    Predict return probability for a single order
    Returns: dict with prediction, probability, risk level
    """
    input_df    = preprocess_input(data, feature_cols)
    probability = float(model.predict_proba(input_df)[0][1])
    prediction  = int(probability >= 0.5)
    risk_level  = get_risk_level(probability)

    return {
        "is_return"          : prediction,
        "return_probability" : round(probability, 4),
        "risk_level"         : risk_level
    }


def predict_batch(
    model,
    data_list: list,
    feature_cols: list
) -> list:
    """
    Predict return probability for multiple orders at once
    Vectorized — processes all orders in one shot!
    Returns: list of prediction dicts
    """
    # Build one DataFrame from all orders
    input_df      = pd.DataFrame(data_list)
    input_df      = input_df.drop(columns=[
        c for c in ['total_delivery_days', 'delivery_delay_days',
                    'is_late_delivery', 'days_to_ship']
        if c in input_df.columns
    ])
    input_df      = input_df[feature_cols]

    # Predict all at once
    probabilities = model.predict_proba(input_df)[:, 1]
    predictions   = (probabilities >= 0.5).astype(int)

    results = []
    for prob, pred in zip(probabilities, predictions):
        results.append({
            "is_return"          : int(pred),
            "return_probability" : round(float(prob), 4),
            "risk_level"         : get_risk_level(float(prob))
        })

    return results


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  PREDICTION PIPELINE TEST")
    print("=" * 50)

    # Load artifacts
    model        = load_model()
    feature_cols = load_feature_cols()

    # Test single prediction
    sample_order = {
        "price"                      : 299.99,
        "freight_value"              : 25.50,
        "product_name_lenght"        : 45.0,
        "product_description_lenght" : 500.0,
        "product_photos_qty"         : 3.0,
        "product_weight_g"           : 800.0,
        "product_length_cm"          : 30.0,
        "product_height_cm"          : 20.0,
        "product_width_cm"           : 15.0,
        "review_score"               : 4.5,
        "freight_ratio"              : 0.085,
        "total_order_value"          : 325.49,
        "product_volume_cm3"         : 9000.0,
        "price_per_gram"             : 0.375,
        "is_repeat_customer"         : 1,
        "customer_order_count"       : 3,
        "seller_total_orders"        : 150,
        "seller_avg_review_score"    : 4.2,
        "purchase_month"             : 11,
        "purchase_dayofweek"         : 4,
        "is_weekend_purchase"        : 0,
        "purchase_hour"              : 22,
        "category_freq_encoded"      : 0.045,
        "customer_state_encoded"     : 0.42,
        "seller_state_encoded"       : 0.71
    }

    result = predict_single(model, sample_order, feature_cols)

    print(f"\n📦 Sample Order Prediction:")
    print(f"  is_return          : {result['is_return']}")
    print(f"  return_probability : {result['return_probability']}")
    print(f"  risk_level         : {result['risk_level']}")
    print("=" * 50)
