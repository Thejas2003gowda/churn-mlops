import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
import joblib
import os


def load_data(path="data/processed/telco_featured.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }


def train_all():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("churn-prediction")

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42, eval_metric="logloss"
        ),
    }

    best_model = None
    best_auc = 0
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            # Log parameters
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()
                              if k in ["max_iter", "n_estimators", "max_depth", "learning_rate"]})
            mlflow.log_param("model_type", name)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Print results
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_model = model
                best_name = name

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(X_train.columns.tolist(), "models/feature_names.joblib")
    print(f"\nBest model: {best_name} (AUC-ROC: {best_auc:.4f})")
    print("Saved to models/best_model.joblib")

    return best_model, best_name


if __name__ == "__main__":
    train_all()