import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Correct Evidently 0.7.x imports
from evidently import Report
from evidently.presets import DataDriftPreset

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/processed_data.csv")

# Drop leakage columns
leakage_cols = [
    'total_delivery_days',
    'delivery_delay_days',
    'is_late_delivery',
    'days_to_ship'
]
df = df.drop(columns=leakage_cols)

X = df.drop('is_return', axis=1)
y = df['is_return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Reference = training data
# Current = test data
reference_data = X_train.copy()
current_data   = X_test.copy()

print(f"✅ Reference data: {reference_data.shape}")
print(f"✅ Current data  : {current_data.shape}")

# ─────────────────────────────────────────
# GENERATE DRIFT REPORT
# ─────────────────────────────────────────
print("\nGenerating Data Drift Report...")

report = Report([DataDriftPreset()])

# run() RETURNS the result — save html on result!
result = report.run(current_data, reference_data)
result.save_html("monitoring/drift_report.html")

print("✅ Drift report saved → monitoring/drift_report.html")
print("Open it in your browser to explore!")