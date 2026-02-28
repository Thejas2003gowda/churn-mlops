from fastapi import FastAPI, HTTPException
from api.schemas import CustomerData, PredictionResponse
import joblib
import pandas as pd
import numpy as np
import os

from src.feature_engineering import engineer_features

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load model and feature names at startup
MODEL_PATH = "models/best_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
THRESHOLD = 0.3  # Optimal from threshold analysis

model = None
feature_names = None


@app.on_event("startup")
def load_model():
    global model, feature_names
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        print(f"Model loaded. Features: {len(feature_names)}")
    else:
        print("WARNING: No model found. Train first.")


@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "no model loaded",
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(500, "Model not loaded")

    # Convert to DataFrame
    data = customer.model_dump(by_alias=True)
    df = pd.DataFrame([data])

    # Engineer features
    df = engineer_features(df)

    # Align columns with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Predict
    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= THRESHOLD)

    if probability >= 0.7:
        risk = "HIGH"
    elif probability >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

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
        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction": int(prob >= THRESHOLD),
            "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
        })

    return {"predictions": results, "count": len(results)}