import pandas as pd
import numpy as np
import pytest

from app import api_fastapi

def test_model_predict_proba():
    # Récupération d'un client
    client_id = int(api_fastapi.data["SK_ID_CURR"].iloc[0])
    X_client = api_fastapi.get_client_features(client_id)

    proba = api_fastapi.pipe.predict_proba(X_client)[0, 1]
    prediction = int(proba >= api_fastapi.threshold)

    # Vérification : proba est valide entre 0 et 1
    assert 0 <= proba <= 1
    # Vérification : prediction est valide entre 0 ou 1
    assert prediction in [0, 1]


def test_imputer_removes_nan():
    # Récupération d'un client
    client_id = int(api_fastapi.data["SK_ID_CURR"].iloc[0])
    X_client = api_fastapi.get_client_features(client_id)

    # Introduction volontaire de NaN
    X_client.iloc[0, 0] = np.nan

    # Transformation avec l'imputer
    X_transformed = api_fastapi.imputer.transform(X_client)

    # Vérification : plus aucun NaN
    assert not np.isnan(X_transformed).any()