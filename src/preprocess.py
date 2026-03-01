import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # Supprimer lignes avec valeurs manquantes
    df = df.dropna()
    # Renommer target en species pour uniformité avec le code existant
    df = df.rename(columns={"target": "species"})
    return df

def save_clean_data(df, output_path):
    df.to_csv(output_path, index=False)