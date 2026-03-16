from pydantic import BaseModel,Field
from typing import Optional

#inout schema this defines what api expects

class OrderInput(BaseModel):
        price: float = Field(..., description="Item price in R$")
        freight_value: float = Field(..., description="Shipping cost in R$")
        product_name_lenght: float = Field(..., description="Product name length")
        product_description_lenght: float = Field(..., description="Product description length")
        product_photos_qty: float = Field(..., description="Number of product photos")
        product_weight_g: float = Field(..., description="Product weight in grams")
        product_length_cm: float = Field(..., description="Product length in cm")
        product_height_cm: float = Field(..., description="Product height in cm")
        product_width_cm: float = Field(..., description="Product width in cm")
        review_score: float = Field(..., description="Customer review score 1-5")
        freight_ratio: float = Field(..., description="Freight cost / price ratio")
        total_order_value: float = Field(..., description="Price + freight total")
        product_volume_cm3: float = Field(..., description="Product volume in cm3")
        price_per_gram: float = Field(..., description="Price divided by weight")
        is_repeat_customer: int = Field(..., description="1 if repeat customer else 0")
        customer_order_count: int = Field(..., description="Total orders by customer")
        seller_total_orders: int = Field(..., description="Total orders by seller")
        seller_avg_review_score: float = Field(..., description="Seller average review")
        purchase_month: int = Field(..., description="Month of purchase 1-12")
        purchase_dayofweek: int = Field(..., description="Day of week 0-6")
        is_weekend_purchase: int = Field(..., description="1 if weekend else 0")
        purchase_hour: int = Field(..., description="Hour of purchase 0-23")
        category_freq_encoded: float = Field(..., description="Category frequency encoding")
        customer_state_encoded: float = Field(..., description="Customer state frequency")
        seller_state_encoded: float = Field(..., description="Seller state frequency")

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
#output schema
class PredictionOutput(BaseModel):
    is_return: int = Field(..., description="1 = likely return, 0 = likely delivered")
    return_probability: float = Field(..., description="Probability of return 0-1")
    risk_level: str = Field(..., description="Low / Medium / High")

class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    total_orders: int
    high_risk_count: int