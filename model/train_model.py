import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Carregar dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Separar features e target
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Criar modelo
model = RandomForestClassifier()

# 5. Treinar modelo
model.fit(X_train, y_train)

# 6. Fazer previsões
predictions = model.predict(X_test)

# 7. Avaliar modelo
print("Model evaluation:")
print(classification_report(y_test, predictions))

# 8. Salvar modelo treinado
joblib.dump(model, "model/fraud_model.pkl")

print("Model saved at model/fraud_model.pkl")
