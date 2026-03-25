import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# Chargement
BASE_DIR = os.path.dirname(__file__)
# écriture des chemins de fichiers en fonction de la base réelle
model_path = os.path.join(BASE_DIR, "pipeline_lgbm.joblib")
threshold_path = os.path.join(BASE_DIR, "threshold_lgbm.joblib")
data_path = os.path.join(BASE_DIR, "app_data.joblib")

model = joblib.load(model_path) # pipeline
threshold = joblib.load(threshold_path) # seuil optimimum def par le modèle
data = joblib.load(data_path) # data

# sélection du client
client_id = st.selectbox(
    "Choisir un client",
    data["SK_ID_CURR"]
)

client_data = data[data["SK_ID_CURR"] == client_id]

# Uniquement les variables explicatives(ID et index ne sont pas des valeurs prédictives)
feats = [f for f in data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X_client = client_data[feats]
