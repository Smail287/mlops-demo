import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from src.preprocess import load_data, clean_data

df = load_data("data/iris.csv")
df = clean_data(df)

X = df.drop("species", axis=1)
y = df["species"]

# Charger le modèle
model = joblib.load("model.pkl")

# Prédiction
y_pred = model.predict(X)

# Vérification
accuracy = accuracy_score(y, y_pred)
assert accuracy > 0.7, "Accuracy too low"
print("Test passed! Accuracy:", accuracy)