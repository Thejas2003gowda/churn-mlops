from pydantic import BaseModel, Field
from typing import Optional


class CustomerData(BaseModel):
    gender: int = Field(ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: int = Field(ge=0, le=1)
    Dependents: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0)
    PhoneService: int = Field(ge=0, le=1)
    MultipleLines: int = Field(ge=0, le=1)
    OnlineSecurity: int = Field(ge=0, le=1)
    OnlineBackup: int = Field(ge=0, le=1)
    DeviceProtection: int = Field(ge=0, le=1)
    TechSupport: int = Field(ge=0, le=1)
    StreamingTV: int = Field(ge=0, le=1)
    StreamingMovies: int = Field(ge=0, le=1)
    PaperlessBilling: int = Field(ge=0, le=1)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    InternetService_Fiber_optic: int = Field(ge=0, le=1, alias="InternetService_Fiber optic", default=0)
    InternetService_No: int = Field(default=0)
    Contract_One_year: int = Field(ge=0, le=1, alias="Contract_One year", default=0)
    Contract_Two_year: int = Field(ge=0, le=1, alias="Contract_Two year", default=0)
    PaymentMethod_Credit_card: int = Field(alias="PaymentMethod_Credit card (automatic)", default=0)
    PaymentMethod_Electronic_check: int = Field(alias="PaymentMethod_Electronic check", default=0)
    PaymentMethod_Mailed_check: int = Field(alias="PaymentMethod_Mailed check", default=0)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    threshold_used: float