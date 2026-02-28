import pandas as pd
from src.feature_engineering import engineer_features


def test_new_features_created():
    df = pd.DataFrame({
        "tenure": [5, 30],
        "TotalCharges": [350.0, 2100.0],
        "MonthlyCharges": [70.0, 70.0],
        "OnlineSecurity": [0, 1],
        "TechSupport": [0, 1],
        "PhoneService": [1, 1],
        "MultipleLines": [0, 1],
        "OnlineBackup": [0, 1],
        "DeviceProtection": [0, 0],
        "StreamingTV": [0, 1],
        "StreamingMovies": [0, 1],
    })
    result = engineer_features(df)
    assert "tenure_group" in result.columns
    assert "avg_monthly_charges" in result.columns
    assert "service_count" in result.columns
    assert "is_new_customer" in result.columns
    assert result["is_new_customer"].tolist() == [1, 0]


def test_service_count():
    df = pd.DataFrame({
        "tenure": [12],
        "TotalCharges": [840.0],
        "MonthlyCharges": [70.0],
        "OnlineSecurity": [1],
        "TechSupport": [1],
        "PhoneService": [1],
        "MultipleLines": [0],
        "OnlineBackup": [0],
        "DeviceProtection": [0],
        "StreamingTV": [0],
        "StreamingMovies": [0],
    })
    result = engineer_features(df)
    assert result["service_count"].iloc[0] == 3