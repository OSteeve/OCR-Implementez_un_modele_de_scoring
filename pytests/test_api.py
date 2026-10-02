from fastapi.testclient import TestClient
from app.api_fastapi import app, data

# Création d'un client de test permettant d'interroger l'API
# sans avoir à lancer réellement le serveur FastAPI
client = TestClient(app)


def test_get_clients():
    """Teste l'endpoint GET /clients."""

    # Envoi d'une requête GET pour récupérer la liste des clients
    response = client.get("/clients")

    # Vérifie que la requête a réussi (HTTP 200)
    assert response.status_code == 200

    # Conversion de la réponse JSON en objet Python
    clients = response.json()

    # Vérifie que la réponse est une liste
    assert isinstance(clients, list)

    # Vérifie que la liste contient au moins un client
    assert len(clients) > 0

    # Vérifie que tous les identifiants clients sont des entiers
    assert all(isinstance(x, int) for x in clients)


def test_predict():
    """Teste l'endpoint POST /predict."""

    # on prend un vrai client existant dans les données
    client_id = int(data["SK_ID_CURR"].iloc[0])

    # Envoi de l'identifiant à l'API pour obtenir une prédiction
    response = client.post("/predict", json={"SK_ID_CURR": client_id})

    # Vérifie que la requête a réussi
    assert response.status_code == 200

    # Récupération du résultat renvoyé par l'API
    result = response.json()

    # Vérifie que toutes les informations attendues sont présentes
    assert "SK_ID_CURR" in result
    assert "proba" in result
    assert "prediction" in result
    assert "threshold" in result

    # Vérifie que l'identifiant retourné correspond au client demandé
    assert result["SK_ID_CURR"] == client_id

    # Vérifie que la prédiction est un entier binaire : 0 ou 1
    assert isinstance(result["prediction"], int)
    assert result["prediction"] in [0, 1]

    # Vérifie que la probabilité est comprise entre 0 et 1
    assert 0 <= result["proba"] <= 1

    # Vérifie que le seuil de décision est compris entre 0 et 1
    assert 0 <= result["threshold"] <= 1


def test_importance():
    """Teste l'endpoint POST /importance."""

    # Sélection d'un vrai client existant
    client_id = int(data["SK_ID_CURR"].iloc[0])

    # Demande à l'API les informations d'explicabilité du modèle
    # pour le client sélectionné
    response = client.post("/importance", json={"SK_ID_CURR": client_id})

    # Vérifie que la requête a réussi
    assert response.status_code == 200

    # Récupération de la réponse JSON
    result = response.json()

    # Vérifie la présence des éléments nécessaires à Shap
    assert "SK_ID_CURR" in result
    assert "shap_values" in result
    assert "feature_names" in result
    assert "feature_values" in result
    assert "base_value" in result

    # Vérifie que l'identifiant retourné est celui demandé
    assert result["SK_ID_CURR"] == client_id

    # Vérifie le type des différentes informations retournées
    assert isinstance(result["shap_values"], list)
    assert isinstance(result["feature_names"], list)
    assert isinstance(result["feature_values"], list)
    assert isinstance(result["base_value"], float)

    # cohérence dimensionnelle
    # chaque variable doit avoir une valeur et une valeur SHAP associées
    assert len(result["shap_values"]) == len(result["feature_names"])
    assert len(result["feature_values"]) == len(result["feature_names"])