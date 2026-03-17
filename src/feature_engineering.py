import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all timestamp columns to datetime"""
    print("Converting timestamps...")

    timestamp_cols = [
        'order_purchase_timestamp',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    print("  ✅ Timestamps converted!")
    return df


def create_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create delivery related features"""
    print("Creating delivery features...")

    df['delivery_delay_days'] = (
        df['order_delivered_customer_date'] -
        df['order_estimated_delivery_date']
    ).dt.days

    df['is_late_delivery'] = (df['delivery_delay_days'] > 0).astype(int)

    df['days_to_ship'] = (
        df['order_delivered_carrier_date'] -
        df['order_purchase_timestamp']
    ).dt.days

    df['total_delivery_days'] = (
        df['order_delivered_customer_date'] -
        df['order_purchase_timestamp']
    ).dt.days

    print("  ✅ Delivery features created!")
    return df


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price and product related features"""
    print("Creating price features...")

    df['freight_ratio']      = df['freight_value'] / (df['price'] + 1)
    df['total_order_value']  = df['price'] + df['freight_value']
    df['product_volume_cm3'] = (
        df['product_length_cm'] *
        df['product_height_cm'] *
        df['product_width_cm']
    )
    df['price_per_gram'] = df['price'] / (df['product_weight_g'] + 1)

    print("  ✅ Price features created!")
    return df


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer and seller features"""
    print("Creating customer features...")

    # Customer order count
    customer_counts = df.groupby(
        'customer_unique_id')['order_id'].count().reset_index()
    customer_counts.columns = ['customer_unique_id', 'customer_order_count']
    df = df.merge(customer_counts, on='customer_unique_id', how='left')

    # Is repeat customer
    df['is_repeat_customer'] = (df['customer_order_count'] > 1).astype(int)

    # Seller total orders
    seller_counts = df.groupby(
        'seller_id')['order_id'].count().reset_index()
    seller_counts.columns = ['seller_id', 'seller_total_orders']
    df = df.merge(seller_counts, on='seller_id', how='left')

    # Seller avg review score
    seller_reviews = df.groupby(
        'seller_id')['review_score'].mean().reset_index()
    seller_reviews.columns = ['seller_id', 'seller_avg_review_score']
    df = df.merge(seller_reviews, on='seller_id', how='left')

    print("  ✅ Customer features created!")
    return df


def create_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create purchase timing features"""
    print("Creating timing features...")

    df['purchase_month']      = df['order_purchase_timestamp'].dt.month
    df['purchase_dayofweek']  = df['order_purchase_timestamp'].dt.dayofweek
    df['is_weekend_purchase'] = (df['purchase_dayofweek'] >= 5).astype(int)
    df['purchase_hour']       = df['order_purchase_timestamp'].dt.hour

    print("  ✅ Timing features created!")
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency encode categorical features"""
    print("Encoding categorical features...")

    # Frequency encode
    category_freq = df['product_category_name'].value_counts() / len(df)
    df['category_freq_encoded'] = df['product_category_name'].map(category_freq)

    state_freq = df['customer_state'].value_counts() / len(df)
    df['customer_state_encoded'] = df['customer_state'].map(state_freq)

    seller_state_freq = df['seller_state'].value_counts() / len(df)
    df['seller_state_encoded'] = df['seller_state'].map(seller_state_freq)

    print("  ✅ Categorical features encoded!")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values"""
    print("Handling missing values...")

    # Fill numerical with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical with mode
    df['product_category_name'] = df['product_category_name'].fillna(
        df['product_category_name'].mode()[0])

    print("  ✅ Missing values handled!")
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw columns and keep only engineered features"""
    print("Selecting final features...")

    cols_to_drop = [
        'order_id', 'customer_id', 'customer_unique_id',
        'product_id', 'seller_id', 'order_item_id',
        'order_purchase_timestamp', 'order_delivered_carrier_date',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'product_category_name', 'customer_state', 'seller_state'
    ]

    # Only drop columns that exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    print(f"  ✅ Final shape: {df.shape}")
    return df


def run_feature_engineering(
    input_path: str  = "data/processed/merged_data.csv",
    output_path: str = "data/processed/processed_data.csv"
) -> pd.DataFrame:
    """Run complete feature engineering pipeline"""
    print("=" * 50)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    df = pd.read_csv(input_path)
    print(f"✅ Loaded: {df.shape}")

    df = convert_timestamps(df)
    df = create_delivery_features(df)
    df = create_price_features(df)
    df = create_customer_features(df)
    df = create_timing_features(df)
    df = encode_categorical_features(df)
    df = handle_missing_values(df)
    df = select_final_features(df)

    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_feature_engineering() 
