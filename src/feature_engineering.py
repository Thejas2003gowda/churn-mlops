import pandas as pd
import numpy as np


def engineer_features(df):
    """Create domain-specific features for churn prediction."""

    # 1. Tenure groups (customer lifecycle stage)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 9999],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # 2. Average monthly spend (spending velocity)
    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # 3. Has premium support (security or tech support)
    df["has_premium_support"] = (
        (df["OnlineSecurity"] == 1) | (df["TechSupport"] == 1)
    ).astype(int)

    # 4. Total service count
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    df["service_count"] = df[service_cols].sum(axis=1)

    # 5. Is new customer (< 6 months)
    df["is_new_customer"] = (df["tenure"] < 6).astype(int)

    # 6. Charge per service (value efficiency)
    df["charge_per_service"] = np.where(
        df["service_count"] > 0,
        df["MonthlyCharges"] / df["service_count"],
        df["MonthlyCharges"]
    )

    # 7. High spender flag
    df["is_high_spender"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_clean.csv")
    print(f"Before: {df.shape[1]} columns")

    df = engineer_features(df)
    print(f"After: {df.shape[1]} columns")

    new_cols = ["tenure_group", "avg_monthly_charges", "has_premium_support",
                "service_count", "is_new_customer", "charge_per_service", "is_high_spender"]
    print(f"\nNew features:")
    for col in new_cols:
        print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")

    df.to_csv("data/processed/telco_featured.csv", index=False)
    print(f"\nSaved to data/processed/telco_featured.csv")