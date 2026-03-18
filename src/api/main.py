from fastapi import FastAPI

from src.inference.predictor import FraudPredictor
from src.api.schemas import Transaction, PredictionResponse


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Machine Learning API for fraud detection"
)

predictor = FraudPredictor()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):

    result = predictor.predict(transaction.dict())

    return result