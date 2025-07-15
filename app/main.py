# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and pipeline
model = joblib.load("models/logistic_regression_model.joblib")
pipeline = joblib.load("models/preprocessing_pipeline.joblib")

# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is live!"}

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])  # Convert to proper shape
    input_transformed = pipeline.transform(input_df)
    
    pred = model.predict(input_transformed)[0]
    prob = model.predict_proba(input_transformed)[0][1]
    
    return {
        "churn_prediction": int(pred),
        "churn_probability": round(prob, 4)
    }
