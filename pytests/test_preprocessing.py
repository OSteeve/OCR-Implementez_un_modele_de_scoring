
import pandas as pd
import pytest

from app.api_fastapi import get_client_features, data


def test_get_client_features_returns_dataframe():
    # récupération du client et ses features
    client_id = int(data["SK_ID_CURR"].iloc[0])

    X_client = get_client_features(client_id)

    # Vérifie que la fonction retourne bien un DataFrame
    assert isinstance(X_client, pd.DataFrame)
    # Vérification de la présence d'une unique ligne (un seul client)
    assert X_client.shape[0] == 1
    # Vérification de l'absence de l'ID dans les features
    assert "SK_ID_CURR" not in X_client.columns
    # Vérification de l'absence de la cible dans les features
    assert "TARGET" not in X_client.columns
