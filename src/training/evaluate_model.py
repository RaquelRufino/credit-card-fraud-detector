import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.pkl"


def load_data(path: str) -> pd.DataFrame:
    """Load dataset."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Split dataset into train and test."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def load_model(path: str):
    """Load trained model."""
    return joblib.load(path)


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))


def main():

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()