from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────
# INPUT SCHEMA WITH PROPER VALIDATION
# ─────────────────────────────────────────

class OrderInput(BaseModel):
    price: float = Field(..., ge=0, le=100000,
                        description="Item price in R$")
    freight_value: float = Field(..., ge=0, le=10000,
                        description="Shipping cost in R$")
    product_name_lenght: float = Field(..., ge=0, le=1000,
                        description="Product name length")
    product_description_lenght: float = Field(..., ge=0, le=100000,
                        description="Product description length")
    product_photos_qty: float = Field(..., ge=0, le=50,
                        description="Number of product photos")
    product_weight_g: float = Field(..., ge=0, le=100000,
                        description="Product weight in grams")
    product_length_cm: float = Field(..., ge=0, le=500,
                        description="Product length in cm")
    product_height_cm: float = Field(..., ge=0, le=500,
                        description="Product height in cm")
    product_width_cm: float = Field(..., ge=0, le=500,
                        description="Product width in cm")
    review_score: float = Field(..., ge=1, le=5,
                        description="Review score 1-5")
    freight_ratio: float = Field(..., ge=0, le=100,
                        description="Freight cost / price ratio")
    total_order_value: float = Field(..., ge=0, le=200000,
                        description="Price + freight total")
    product_volume_cm3: float = Field(..., ge=0, le=10000000,
                        description="Product volume in cm3")
    price_per_gram: float = Field(..., ge=0, le=10000,
                        description="Price divided by weight")
    is_repeat_customer: int = Field(..., ge=0, le=1,
                        description="1 if repeat customer else 0")
    customer_order_count: int = Field(..., ge=1, le=1000,
                        description="Total orders by customer")
    seller_total_orders: int = Field(..., ge=1, le=100000,
                        description="Total orders by seller")
    seller_avg_review_score: float = Field(..., ge=1, le=5,
                        description="Seller average review")
    purchase_month: int = Field(..., ge=1, le=12,
                        description="Month 1-12")
    purchase_dayofweek: int = Field(..., ge=0, le=6,
                        description="Day 0=Monday 6=Sunday")
    is_weekend_purchase: int = Field(..., ge=0, le=1,
                        description="1 if weekend else 0")
    purchase_hour: int = Field(..., ge=0, le=23,
                        description="Hour 0-23")
    category_freq_encoded: float = Field(..., ge=0, le=1,
                        description="Category frequency encoding")
    customer_state_encoded: float = Field(..., ge=0, le=1,
                        description="Customer state frequency")
    seller_state_encoded: float = Field(..., ge=0, le=1,
                        description="Seller state frequency")

    class Config:
        json_schema_extra = {
            "example": {
                "price": 299.99,
                "freight_value": 25.50,
                "product_name_lenght": 45.0,
                "product_description_lenght": 500.0,
                "product_photos_qty": 3.0,
                "product_weight_g": 800.0,
                "product_length_cm": 30.0,
                "product_height_cm": 20.0,
                "product_width_cm": 15.0,
                "review_score": 3.5,
                "freight_ratio": 0.085,
                "total_order_value": 325.49,
                "product_volume_cm3": 9000.0,
                "price_per_gram": 0.375,
                "is_repeat_customer": 0,
                "customer_order_count": 1,
                "seller_total_orders": 150,
                "seller_avg_review_score": 3.8,
                "purchase_month": 11,
                "purchase_dayofweek": 4,
                "is_weekend_purchase": 0,
                "purchase_hour": 22,
                "category_freq_encoded": 0.045,
                "customer_state_encoded": 0.42,
                "seller_state_encoded": 0.71
            }
        }

# ─────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────
class PredictionOutput(BaseModel):
    is_return: int = Field(..., description="1=likely return 0=delivered")
    return_probability: float = Field(..., description="Probability 0-1")
    risk_level: str = Field(..., description="Low/Medium/High")

class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    total_orders: int
    high_risk_count: int