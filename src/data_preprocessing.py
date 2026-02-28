import pandas as pd
import numpy as np
import os


def load_raw_data(path="data/raw/telco_churn.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Handle missing values and fix data types."""
    # TotalCharges has blank strings — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges (11 rows)
    df = df.dropna(subset=["TotalCharges"])

    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    return df


def encode_features(df):
    """Encode categorical features."""
    # Binary columns: Yes/No -> 1/0
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    for col in binary_cols:
        if col == "gender":
            df[col] = df[col].map({"Male": 1, "Female": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # SeniorCitizen is already 0/1

    # Multi-value columns with "No internet/phone service" -> "No"
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        df[col] = df[col].replace("No phone service", "No").replace("No internet service", "No")
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    return df


def preprocess(input_path="data/raw/telco_churn.csv", output_path="data/processed/telco_clean.csv"):
    """Full preprocessing pipeline."""
    df = load_raw_data(input_path)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_data(df)
    print(f"After cleaning: {df.shape[0]} rows")

    df = encode_features(df)
    print(f"After encoding: {df.shape[1]} columns")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    df = preprocess()
    print(f"\nTarget distribution:\n{df['Churn'].value_counts()}")
    print(f"\nChurn rate: {df['Churn'].mean():.1%}")