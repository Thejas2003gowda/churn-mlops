from fastapi import FastAPI, HTTPException, Response
from api.schemas import CustomerData, PredictionResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import joblib
import pandas as pd
import numpy as np
import os
import time

from src.feature_engineering import engineer_features

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions made")
CHURN_PREDICTED = Counter("churn_predicted_total", "Predictions by label", ["prediction"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction response time",
                                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
PREDICTION_PROBABILITY = Histogram("prediction_probability", "Churn probability distribution",
                                    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MODEL_INFO = Gauge("model_loaded", "Whether model is loaded", ["model_path", "threshold"])

# Load model
MODEL_PATH = "models/best_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
THRESHOLD = 0.3

model = None
feature_names = None


@app.on_event("startup")
def load_model():
    global model, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        MODEL_INFO.labels(model_path=MODEL_PATH, threshold=str(THRESHOLD)).set(1)
        print(f"Model loaded. Features: {len(feature_names)}")
    else:
        MODEL_INFO.labels(model_path=MODEL_PATH, threshold=str(THRESHOLD)).set(0)
        print("WARNING: No model found.")


@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "no model loaded",
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(500, "Model not loaded")

    start_time = time.time()

    data = customer.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    df = engineer_features(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= THRESHOLD)

    if probability >= 0.7:
        risk = "HIGH"
    elif probability >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Record metrics
    latency = time.time() - start_time
    PREDICTIONS_TOTAL.inc()
    CHURN_PREDICTED.labels(prediction="churn" if prediction == 1 else "retain").inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_PROBABILITY.observe(probability)

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        risk_level=risk,
        threshold_used=THRESHOLD,
    )


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    if model is None:
        raise HTTPException(500, "Model not loaded")

    results = []
    for c in customers:
        data = c.model_dump(by_alias=True)
        df = pd.DataFrame([data])
        df = engineer_features(df)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= THRESHOLD)

        PREDICTIONS_TOTAL.inc()
        CHURN_PREDICTED.labels(prediction="churn" if pred == 1 else "retain").inc()
        PREDICTION_PROBABILITY.observe(prob)

        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction": pred,
            "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
        })

    return {"predictions": results, "count": len(results)}