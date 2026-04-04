from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os
import numpy as np

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Chargement du modèle et des ressources
pipe_path = os.path.join(BASE_DIR, "pipe_lgbm.joblib")
pipe = joblib.load(pipe_path)

threshold_path = os.path.join(BASE_DIR, "threshold_lgbm.joblib")
threshold = joblib.load(threshold_path)

data_path = os.path.join(BASE_DIR, "app_data.joblib")
data = joblib.load(data_path)

# Extraction du modèle et imputer du pipeline
model = pipe.named_steps["model"]
imputer = pipe.named_steps["imputer"]

explainer = shap.TreeExplainer(model)

app = FastAPI()

class ClientRequest(BaseModel):
    SK_ID_CURR: int

def get_client_features(client_id: int) -> pd.DataFrame:
    # selection du client
    client_data = data[data["SK_ID_CURR"] == client_id].copy()

    # Erreur si introuvable
    if client_data.empty:
        raise ValueError(f"Client {client_id} introuvable")
    
    # suppression des colonnes non explicatives
    feats = [
        f for f in data.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    # Récupération des data du clients
    X_client = client_data[feats].copy()
   
    return X_client

@app.get("/clients")
def get_clients():
    return data["SK_ID_CURR"].tolist()


@app.post("/predict")
def predict(request: ClientRequest):
    # features du client selectionné
    X_client = get_client_features(request.SK_ID_CURR)
    
    proba = pipe.predict_proba(X_client)[0, 1]
    prediction = int(proba >= threshold)

    return {
        "SK_ID_CURR": request.SK_ID_CURR,
        "proba": float(proba),
        "prediction": prediction,
        "threshold": float(threshold)
    }

@app.post("/importance")
def importance(request: ClientRequest):
    # features du client selectionné
    X_client = get_client_features(request.SK_ID_CURR)

    X_client_transformed = imputer.transform(X_client)
    shap_values = explainer(X_client_transformed)

    feature_names = [str(col) for col in X_client.columns.tolist()]

    feature_values = X_client_transformed[0].tolist()
    
    base_value = float(shap_values.base_values[0])

    return {
        "SK_ID_CURR": request.SK_ID_CURR,
        "shap_values": shap_values.values[0].tolist(),
        "feature_names": feature_names,
        "feature_values": feature_values,
        "base_value":base_value
    }
