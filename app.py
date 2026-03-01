import streamlit as st
import joblib
import pandas as pd

st.title("Démo MLOps - Iris Classifier")

# Charger modèle
model = joblib.load("model.pkl")

# Interface simple
sepal_length = st.slider("Sepal length", 4.0, 8.0)
sepal_width = st.slider("Sepal width", 2.0, 4.5)
petal_length = st.slider("Petal length", 1.0, 7.0)
petal_width = st.slider("Petal width", 0.1, 2.5)

df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                  columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

prediction = model.predict(df)[0]
st.write("Prediction:", prediction)