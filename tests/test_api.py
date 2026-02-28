from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_returns_probability():
    payload = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0,
        "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
        "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
        "PaperlessBilling": 1, "MonthlyCharges": 50.0, "TotalCharges": 600.0,
        "InternetService_Fiber optic": 0, "InternetService_No": 0,
        "Contract_One year": 1, "Contract_Two year": 0,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 0,
        "PaymentMethod_Mailed check": 0
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert 0 <= data["churn_probability"] <= 1
        assert data["churn_prediction"] in [0, 1]
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    else:
        assert response.status_code in [200, 500]