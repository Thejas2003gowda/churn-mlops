import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import joblib
import json
import os


def threshold_analysis(model, X_test, y_test):
    """Evaluate model at multiple classification thresholds."""
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Business cost: assume losing a customer costs $500,
        # retention offer costs $50
        cost = fn * 500 + fp * 50
        saved = tp * 500 - tp * 50  # revenue saved minus offer cost

        results.append({
            "threshold": t,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "missed_churners": int(fn),
            "unnecessary_offers": int(fp),
            "business_cost": int(cost),
            "revenue_saved": int(saved),
        })

    return results


def run_evaluation():
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/processed/telco_featured.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load("models/best_model.joblib")
    y_prob = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC-ROC: {auc_score:.4f}")

    # Threshold analysis
    results = threshold_analysis(model, X_test, y_test)

    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Missed':>10} {'Unnecessary':>12} {'Cost ($)':>10} {'Saved ($)':>10}")
    print("-" * 95)
    for r in results:
        print(f"{r['threshold']:>10.2f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['missed_churners']:>10} {r['unnecessary_offers']:>12} {r['business_cost']:>10,} {r['revenue_saved']:>10,}")

    # Find optimal threshold (minimize business cost)
    best = min(results, key=lambda x: x["business_cost"])
    print(f"\nOptimal threshold: {best['threshold']} (lowest business cost: ${best['business_cost']:,})")

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    output = {
        "auc_roc": auc_score,
        "threshold_analysis": results,
        "optimal_threshold": best["threshold"],
    }
    with open("evaluation/threshold_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to evaluation/threshold_analysis.json")


if __name__ == "__main__":
    run_evaluation()