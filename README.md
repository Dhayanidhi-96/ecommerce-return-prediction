 # 📦 E-Commerce Return Prediction — MLOps Pipeline

> Predicts whether an e-commerce order will be returned **before it ships** — 
> enabling proactive intervention and cost savings.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-red)](https://mlflow.org)

---

## 🎯 Business Problem

E-commerce companies lose **billions annually** on returns — shipping costs, 
restocking fees, and refunds. If a return can be predicted **before shipping**, 
companies can:
- Flag high-risk orders for review
- Offer targeted discounts to retain customers
- Adjust inventory and logistics planning

---

## 🏗️ Architecture
```
Raw Data (Olist Dataset)
        ↓
Data Preprocessing (src/data_preprocessing.py)
        ↓
Feature Engineering (src/feature_engineering.py)
        ↓
Model Training — XGBoost (src/train.py)
        ↓
Experiment Tracking — MLflow
        ↓
REST API — FastAPI (api/main.py)
        ↓
Containerization — Docker
        ↓
Monitoring — Evidently AI
        ↓
Cloud Deployment — AWS EC2
```

---

## ⚙️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| ML Model | XGBoost, LightGBM, Logistic Regression |
| Experiment Tracking | MLflow 3.10 |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Containerization | Docker |
| Monitoring | Evidently AI |
| Cloud | AWS EC2 |
| Package Manager | uv |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| PR-AUC | 0.8134 |
| Recall | 0.6911 |
| F1 Score | 0.8053 |
| Precision | 0.9647 |

> **Why PR-AUC?** With only 1.18% return rate, accuracy is misleading. 
> PR-AUC captures model performance on the minority class correctly.

---

## 📁 Project Structure
```
ecommerce-return-prediction/
├── api/
│   ├── main.py              # FastAPI application
│   └── schema.py            # Pydantic input/output schemas
├── data/
│   ├── raw/                 # Original Olist CSV files
│   └── processed/           # Cleaned & engineered features
├── models/
│   ├── best_model.pkl       # Trained XGBoost model
│   ├── scaler.pkl           # StandardScaler
│   ├── feature_cols.pkl     # Feature column names
│   └── metrics.json         # Model performance metrics
├── monitoring/
│   └── evidently_report.py  # Data drift monitoring
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── data_preprocessing.py # Data loading & cleaning
│   ├── feature_engineering.py # Feature creation
│   ├── train.py              # MLflow experiment tracking
│   └── predict.py            # Prediction pipeline
├── .env                     # Environment variables (not in git)
├── .dockerignore
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+
- uv package manager
- Docker Desktop

### Setup
```bash
# Clone repo
git clone https://github.com/Dhayanidhi-96/ecommerce-return-prediction.git
cd ecommerce-return-prediction

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
```

### Run Full Pipeline
```bash
# Step 1: Preprocessing
uv run python src/data_preprocessing.py

# Step 2: Feature Engineering
uv run python src/feature_engineering.py

# Step 3: Train models (set RUN_EXPERIMENTS=True in train.py)
uv run python src/train.py

# Step 4: Run API
uvicorn api.main:app --reload --port 8000

# Step 5: Run MLflow UI
uv run mlflow ui
```

### Run with Docker
```bash
# Build image
docker build -t return-predictor .

# Run container
docker run -p 8000:8000 return-predictor

# API live at:
# http://localhost:8000/docs
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single order prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/model/info` | Model details & metrics |

### Sample Request
```json
POST /predict
{
    "price": 299.99,
    "freight_value": 25.50,
    "review_score": 3.5,
    "is_repeat_customer": 0,
    "purchase_hour": 22,
    ...
}
```

### Sample Response
```json
{
    "is_return": 1,
    "return_probability": 0.73,
    "risk_level": "High 🔴"
}
```

---

## 📈 MLflow Experiments

5 experiments tracked and compared:

| Model | PR-AUC | Recall | F1 |
|-------|--------|--------|----|
| Logistic Regression | 0.3202 | 0.0987 | 0.1685 |
| XGBoost | **0.8134** | **0.6911** | **0.8053** |
| LightGBM | 0.8104 | 0.7165 | 0.8144 |
| XGBoost + SMOTE | 0.7833 | 0.7089 | 0.7650 |
| LightGBM + SMOTE | 0.7819 | 0.6987 | 0.7635 |

---

## 🔍 Monitoring

Evidently AI monitors data drift between training and production data:
```bash
# Generate drift report
uv run python monitoring/evidently_report.py

# Report saved to:
# monitoring/drift_report.html
```

---

## 🌍 Live Demo

API live at: **http://your-ec2-ip:8000/docs**

---

## 📝 Resume Bullets

- Built production-ready E-Commerce Return Prediction pipeline processing 
  **110,000+ orders** with **<100ms inference latency** using FastAPI and XGBoost
- Engineered **25 behavioral and temporal features** improving PR-AUC to **0.8134**
- Tracked **5 ML experiments** with MLflow — full reproducibility and model versioning
- Monitored production model using **Evidently AI** for real-time data drift detection
- Containerized complete MLOps pipeline with **Docker** and deployed on **AWS EC2**
```

---

Save and then create `.env.example` file:
```
MODEL_PATH=models/best_model.pkl
SCALER_PATH=models/scaler.pkl
FEATURE_COLS_PATH=models/feature_cols.pkl
METRICS_PATH=models/metrics.json
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
