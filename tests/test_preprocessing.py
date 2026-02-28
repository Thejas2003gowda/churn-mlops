import pandas as pd
from src.data_preprocessing import clean_data, encode_features


def test_clean_removes_nulls():
    df = pd.DataFrame({
        "customerID": ["1", "2"],
        "TotalCharges": ["100.5", " "],
        "tenure": [10, 5],
    })
    result = clean_data(df)
    assert result["TotalCharges"].isna().sum() == 0
    assert "customerID" not in result.columns


def test_encode_binary():
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "Churn": ["Yes", "No"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "PaperlessBilling": ["Yes", "No"],
        "MultipleLines": ["Yes", "No"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "InternetService": ["Fiber optic", "DSL"],
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
    })
    result = encode_features(df)
    assert result["gender"].tolist() == [1, 0]
    assert result["Churn"].tolist() == [1, 0]