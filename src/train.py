import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, average_precision_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONTROL WHAT RUNS
# Change these flags to control execution
# ─────────────────────────────────────────
RUN_EXPERIMENTS = False  # Set True only when you want to train
REGISTER_MODEL  = False   # Set True only when you want to register

# ─────────────────────────────────────────
# MLFLOW SETUP
# ─────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ecommerce-return-prediction")
print("✅ MLflow setup complete!")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv("data/processed/processed_data.csv")

leakage_cols = [
    'total_delivery_days',
    'delivery_delay_days',
    'is_late_delivery',
    'days_to_ship'
]

X = df.drop(columns=['is_return'] + leakage_cols)
y = df['is_return']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"✅ Data loaded!")
print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")

# ─────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────
if RUN_EXPERIMENTS:

    # ── Experiment 1: Logistic Regression ──
    print("\nRunning Experiment 1: Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression_baseline"):

        mlflow.log_param("model_name",       "LogisticRegression")
        mlflow.log_param("test_size",         0.3)
        mlflow.log_param("random_state",      42)
        mlflow.log_param("max_iter",          1000)
        mlflow.log_param("smote",             False)
        mlflow.log_param("training_samples",  X_train.shape[0])
        mlflow.log_param("features",          X_train.shape[1])

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)

        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("pr_auc",    average_precision_score(y_test, y_prob))

        mlflow.sklearn.log_model(lr, "model")
        print(f"  ✅ Logged! PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    # ── Experiment 2: XGBoost ──
    print("\nRunning Experiment 2: XGBoost...")
    with mlflow.start_run(run_name="xgboost_no_smote"):

        mlflow.log_param("model_name",      "XGBoost")
        mlflow.log_param("test_size",        0.3)
        mlflow.log_param("random_state",     42)
        mlflow.log_param("n_estimators",     200)
        mlflow.log_param("max_depth",        6)
        mlflow.log_param("learning_rate",    0.1)
        mlflow.log_param("smote",            False)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42,
            eval_metric='aucpr', verbosity=0
        )
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("pr_auc",    average_precision_score(y_test, y_prob))

        mlflow.xgboost.log_model(xgb, "model")
        print(f"  ✅ Logged! PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    # ── Experiment 3: LightGBM ──
    print("\nRunning Experiment 3: LightGBM...")
    with mlflow.start_run(run_name="lightgbm_no_smote"):

        mlflow.log_param("model_name",      "LightGBM")
        mlflow.log_param("test_size",        0.3)
        mlflow.log_param("random_state",     42)
        mlflow.log_param("n_estimators",     200)
        mlflow.log_param("max_depth",        6)
        mlflow.log_param("learning_rate",    0.1)
        mlflow.log_param("smote",            False)

        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)

        y_pred = lgbm.predict(X_test)
        y_prob = lgbm.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("pr_auc",    average_precision_score(y_test, y_prob))

        mlflow.lightgbm.log_model(lgbm, "model")
        print(f"  ✅ Logged! PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    # ── Experiment 4: XGBoost + SMOTE ──
    print("\nRunning Experiment 4: XGBoost + SMOTE...")
    with mlflow.start_run(run_name="xgboost_with_smote"):

        mlflow.log_param("model_name",      "XGBoost+SMOTE")
        mlflow.log_param("test_size",        0.3)
        mlflow.log_param("random_state",     42)
        mlflow.log_param("n_estimators",     200)
        mlflow.log_param("max_depth",        6)
        mlflow.log_param("learning_rate",    0.1)
        mlflow.log_param("smote",            True)

        xgb_smote = XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42,
            eval_metric='aucpr', verbosity=0
        )
        xgb_smote.fit(X_train_smote, y_train_smote)

        y_pred = xgb_smote.predict(X_test)
        y_prob = xgb_smote.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("pr_auc",    average_precision_score(y_test, y_prob))

        mlflow.xgboost.log_model(xgb_smote, "model")
        print(f"  ✅ Logged! PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    # ── Experiment 5: LightGBM + SMOTE ──
    print("\nRunning Experiment 5: LightGBM + SMOTE...")
    with mlflow.start_run(run_name="lightgbm_with_smote"):

        mlflow.log_param("model_name",      "LightGBM+SMOTE")
        mlflow.log_param("test_size",        0.3)
        mlflow.log_param("random_state",     42)
        mlflow.log_param("n_estimators",     200)
        mlflow.log_param("max_depth",        6)
        mlflow.log_param("learning_rate",    0.1)
        mlflow.log_param("smote",            True)

        lgbm_smote = LGBMClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42,
            verbose=-1
        )
        lgbm_smote.fit(X_train_smote, y_train_smote)

        y_pred = lgbm_smote.predict(X_test)
        y_prob = lgbm_smote.predict_proba(X_test)[:, 1]

        mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall",    recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score",  f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("pr_auc",    average_precision_score(y_test, y_prob))

        mlflow.lightgbm.log_model(lgbm_smote, "model")
        print(f"  ✅ Logged! PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

    print("\n🎉 All 5 experiments logged!")

# ─────────────────────────────────────────
# REGISTER BEST MODEL
# ─────────────────────────────────────────
if REGISTER_MODEL:
    print("\nFinding best model...")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("ecommerce-return-prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.pr_auc DESC"]
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_pr_auc = best_run.data.metrics['pr_auc']
    best_model_name = best_run.data.params['model_name']

    print(f"Best model  : {best_model_name}")
    print(f"Best PR-AUC : {best_pr_auc:.4f}")
    print(f"Run ID      : {best_run_id}")

    # Register model
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name="ReturnPredictor"
    )

    print(f"\n✅ Model registered as 'ReturnPredictor'!")
    print(f"Version: {registered.version}")

    # Save model locally too
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X_train.columns.tolist(), "models/feature_cols.pkl")
    print("✅ Scaler and feature cols saved to models/")

# Set alias instead of stage (MLflow 3.x)
    client.set_registered_model_alias(
        name="ReturnPredictor",
        alias="champion",
        version=registered.version
    )
    print(f"✅ Version {registered.version} set as 'champion'!")

    # Test loading from registry using alias
    loaded_model = mlflow.xgboost.load_model(
        "models:/ReturnPredictor@champion"
    )
    test_pred = loaded_model.predict(X_test[:5])
    print(f"✅ Model loads correctly from registry!")
    print(f"Sample predictions: {test_pred}")