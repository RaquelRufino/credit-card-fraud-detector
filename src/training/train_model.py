import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


def train_model(X_train, y_train):
    """Train RandomForest model."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, path: str):
    """Persist trained model."""
    Path("models").mkdir(exist_ok=True)

    joblib.dump(model, path)


def main():

    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Saving model...")
    save_model(model, MODEL_PATH)

    print("Training completed.")


if __name__ == "__main__":
    main()