import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.preprocess import load_data, clean_data

# Charger et nettoyer les données
df = load_data("data/iris.csv")
df = clean_data(df)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "model.pkl")