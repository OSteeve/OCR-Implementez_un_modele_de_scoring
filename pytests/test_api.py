from fastapi.testclient import TestClient
from app.api_fastapi import app, data

client = TestClient(app)


def test_get_clients():
    response = client.get("/clients")

    assert response.status_code == 200

    clients = response.json()

    assert isinstance(clients, list)
    assert len(clients) > 0
    assert all(isinstance(x, int) for x in clients)


def test_predict():
    # on prend un vrai client existant dans les données
    client_id = int(data["SK_ID_CURR"].iloc[0])

    response = client.post("/predict", json={"SK_ID_CURR": client_id})

    assert response.status_code == 200

    result = response.json()

    assert "SK_ID_CURR" in result
    assert "proba" in result
    assert "prediction" in result
    assert "threshold" in result

    assert result["SK_ID_CURR"] == client_id
    assert isinstance(result["prediction"], int)
    assert result["prediction"] in [0, 1]
    assert 0 <= result["proba"] <= 1
    assert 0 <= result["threshold"] <= 1


def test_importance():
    # on prend un vrai client existant
    client_id = int(data["SK_ID_CURR"].iloc[0])

    response = client.post("/importance", json={"SK_ID_CURR": client_id})

    assert response.status_code == 200

    result = response.json()

    assert "SK_ID_CURR" in result
    assert "shap_values" in result
    assert "feature_names" in result
    assert "feature_values" in result
    assert "base_value" in result

    assert result["SK_ID_CURR"] == client_id
    assert isinstance(result["shap_values"], list)
    assert isinstance(result["feature_names"], list)
    assert isinstance(result["feature_values"], list)
    assert isinstance(result["base_value"], float)

    # cohérence dimensionnelle
    assert len(result["shap_values"]) == len(result["feature_names"])
    assert len(result["feature_values"]) == len(result["feature_names"])