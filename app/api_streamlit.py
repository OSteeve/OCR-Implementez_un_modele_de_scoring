import streamlit as st
import requests
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# API_URL = "http://localhost:8000"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Bonjour, bienvenu sur : Scoring risque client")

# Charger les clients via API
clients = requests.get(f"{API_URL}/clients").json()
# sélection du client
client_id = st.selectbox(
    "Selectionner ou taper un ID client",
    clients
)


# Prédiction
if st.button("Prédire"):

    # appel API predict
    response = requests.post(
        f"{API_URL}/predict",
        json={"SK_ID_CURR": int(client_id)}
    )

    result = response.json()

    proba = result["proba"]
    threshold = result["threshold"]

    st.metric("Probabilité de défaut", f"{proba:.3f}")

    st.subheader("Décision")
    if result["prediction"] == 1:
        st.error("## REFUS")
    else:
        st.success("## ACCORD")

    # ------------------------------
    # Jauge
    # ------------------------------
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba,
            number={"valueformat": ".3f"},
            title={"text": "Risque de défaut"},
            gauge={
                "axis": {"range": [0, 1]},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "value": threshold
                }
            }
        )
    )

    st.plotly_chart(fig)


    # SHAP
    # Envois de la requête
    response_shap = requests.post(
        f"{API_URL}/importance",
        json={"SK_ID_CURR": int(client_id)}
    )

    # Récupération du dictionnaire en reponse
    shap_result = response_shap.json()

    st.subheader("Influence des variables")

    # Créeation de l'objet SHAP
    explanation = shap.Explanation(
        values=np.array(shap_result["shap_values"]),
        base_values=shap_result["base_value"],
        data=np.array(shap_result["feature_values"]),
        feature_names=shap_result["feature_names"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=20, show=False)
    st.pyplot(fig)