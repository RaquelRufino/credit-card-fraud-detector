import joblib
import pandas as pd


MODEL_PATH = "models/fraud_model.pkl"


class FraudPredictor:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = joblib.load(model_path)

    def predict(self, transaction: dict):

        df = pd.DataFrame([transaction])

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": round(float(probability), 6)
        }