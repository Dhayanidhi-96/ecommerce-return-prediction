import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(data_path: str = "data/raw/") -> dict:
    """Load all raw CSV files from data/raw/"""
    print("Loading raw datasets...")

    datasets = {
        'orders'      : pd.read_csv(f"{data_path}olist_orders_dataset.csv"),
        'order_items' : pd.read_csv(f"{data_path}olist_order_items_dataset.csv"),
        'products'    : pd.read_csv(f"{data_path}olist_products_dataset.csv"),
        'customers'   : pd.read_csv(f"{data_path}olist_customers_dataset.csv"),
        'reviews'     : pd.read_csv(f"{data_path}olist_order_reviews_dataset.csv"),
        'sellers'     : pd.read_csv(f"{data_path}olist_sellers_dataset.csv"),
    }

    for name, df in datasets.items():
        print(f"  ✅ {name}: {df.shape}")

    return datasets


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Filter completed orders and create binary target"""
    print("\nCreating target variable...")

    # Keep only completed orders
    completed = orders[orders['order_status'].isin([
        'delivered', 'canceled', 'unavailable'
    ])].copy()

    # Create binary target
    completed['is_return'] = completed['order_status'].apply(
        lambda x: 1 if x in ['canceled', 'unavailable'] else 0
    )

    print(f"  ✅ Completed orders: {completed.shape[0]}")
    print(f"  ✅ Return rate: {completed['is_return'].mean()*100:.2f}%")

    return completed


def merge_datasets(datasets: dict) -> pd.DataFrame:
    """Merge all datasets into one master dataframe"""
    print("\nMerging datasets...")

    orders = datasets['orders']

    # Create target first
    df = create_target(orders)

    # Merge all tables
    df = df.merge(datasets['order_items'], on='order_id',   how='left')
    df = df.merge(datasets['products'],    on='product_id', how='left')
    df = df.merge(datasets['customers'],   on='customer_id', how='left')
    df = df.merge(datasets['reviews'],     on='order_id',   how='left')
    df = df.merge(datasets['sellers'],     on='seller_id',  how='left')

    print(f"  ✅ Merged shape: {df.shape}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged dataframe"""
    print("\nCleaning data...")

    # Drop useless columns
    cols_to_drop = [
        'review_comment_title',
        'review_comment_message',
        'review_id',
        'review_creation_date',
        'review_answer_timestamp',
        'customer_city',
        'seller_city',
        'order_status',
        'customer_zip_code_prefix',
        'seller_zip_code_prefix',
        'shipping_limit_date',
        'order_approved_at'
    ]

    # Only drop columns that exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  ✅ Removed {before - len(df)} duplicates")

    print(f"  ✅ Clean shape: {df.shape}")

    return df


def run_preprocessing(
    data_path: str = "data/raw/",
    save_path: str = "data/processed/merged_data.csv"
) -> pd.DataFrame:
    """Run complete preprocessing pipeline"""
    print("=" * 50)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 50)

    datasets = load_raw_data(data_path)
    df       = merge_datasets(datasets)
    df       = clean_data(df)

    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved to {save_path}")
    print("=" * 50)

    return df


if __name__ == "__main__":
    run_preprocessing()
