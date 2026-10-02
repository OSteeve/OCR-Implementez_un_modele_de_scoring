# Projet OCR — Implémentez un modèle de scoring

## Présentation du projet

Ce projet a pour objectif de développer un **modèle de scoring de crédit** permettant d'estimer la probabilité qu'un client rencontre des difficultés de remboursement.

Il s'appuie sur les données du challenge **Home Credit Default Risk**.

---

## Objectifs

Les principales étapes du projet sont :

- explorer et préparer les données clients ;
- construire les variables utiles à la modélisation ;
- entraîner et comparer plusieurs modèles de classification ;
- générer un tracking d'expérimentations à l’aide de MLFlow ;
- prendre en compte le déséquilibre entre les classes ;
- définir un seuil de décision adapté au contexte métier ;
- interpréter les prédictions du modèle avec SHAP ;
- mettre le modèle à disposition via une API FastAPI ;
- développer une interface utilisateur avec Streamlit ;
- tester l'API avec Pytest ;
- conteneuriser l'application avec Docker ;
- automatiser les tests et le déploiement avec GitHub Actions ;
- déployer l'application sur une instance AWS EC2.

---

Organisation du dépôt

``` text
OCR-Implementez_un_modele_de_scoring/
│
├── .devcontainer/
│   └── devcontainer.json       # Configuration d'un conteneur
│       └── ...
│
├── .github/workflows
│   └── deploy.yml
│       └── ...                 # Pipeline CI/CD GitHub Actions
│
├── app/
│   ├── api_fastapi.py          # API de scoring FastAPI
│   ├── api_streamlit.py        # Interface utilisateur Streamlit
│   ├── Dockerfile              # Construction de l'image Docker
│   ├── pipe_lgbm.joblib        # Pipeline du modèle entraîné
│   ├── threshold_lgbm.joblib   # Seuil de décision métier
│   └── ...                     # Ressources nécessaires à l'application
│
├── Notebook/
│   ├── Otto_Steeve_2_notebook.. # Exploration, préparation et modélisation
│   ├── streamlit_script.py     # script de test de streamlit et du conteneur
    └── ...                     
│
├── pytests/
│   └── ...                     # Tests automatisés de l'API
│
├── requirements.txt            # Dépendances Python
│
└── README.md                   # Présentation du projet

````
## Modèle de scoring

Le modèle estime pour chaque client une **probabilité de défaut de paiement**.

Cette probabilité est comparée à un seuil de décision défini lors de la phase de modélisation.

```text
Probabilité de défaut
        │
        ▼
Comparaison au seuil métier
        │
        ├── Risque < seuil  → Crédit accordé
        │
        └── Risque ≥ seuil  → Crédit refusé
```

Le modèle retenu repose sur **LightGBM**.

---

## Suivi des expérimentations avec MLflow

Les expérimentations de Machine Learning sont suivies avec **MLflow**.

MLflow permet de centraliser et de comparer les différents essais réalisés pendant la phase de modélisation, notamment :

- les modèles testés ;
- les hyperparamètres utilisés ;
- les métriques de performance ;
- les temps d'exécution
- les résultats obtenus pour chaque expérimentation.

Cette approche facilite la comparaison des modèles et permet de conserver une trace des différentes expérimentations ayant conduit au choix du modèle final **LightGBM**.

---

## Architecture de l'application

L'application repose sur une séparation entre l'interface utilisateur et le moteur de scoring.

```text
Utilisateur
        ↓
Streamlit (api_streamlit.py)  
    → Requêtes HTTP  
        ↓  
FastAPI (api_fastapi.py)  
        ↓  
Pipeline de Machine Learning   
    → Prétraitement  
    → Modèle LightGBM  
    → Seuil de décision  
  ```
  
**Streamlit** constitue l'interface utilisateur.

**FastAPI** centralise la logique de prédiction et permet à Streamlit d'interroger le modèle via des requêtes HTTP.

---

## Organisation du dépôt

```text
OCR-Implementez_un_modele_de_scoring/
│
├── .github/
│   └── workflows/
│       └── ...                 # Pipeline CI/CD GitHub Actions
│
├── .devcontainer/
│   └── devcontainer.json       # Configuration de l'environnement de développement
│
├── app/
│   ├── api_fastapi.py          # API de scoring FastAPI
│   ├── api_streamlit.py        # Interface utilisateur Streamlit
│   ├── Dockerfile              # Construction de l'image Docker
│   ├── pipe_lgbm.joblib        # Pipeline du modèle entraîné
│   ├── threshold_lgbm.joblib   # Seuil de décision métier
│   └── ...                     # Ressources nécessaires à l'application
│
├── Notebook/
│   └── Notebook                # Exploration, préparation et modélisation
│
├── pytests/
│   └── ...                     # Tests automatisés de l'API
│
├── requirements.txt            # Dépendances Python
│
└── README.md                   # Présentation du projet
```


---

## Interprétation avec SHAP

Afin de rendre les prédictions du modèle plus compréhensibles, l'application utilise **SHAP (SHapley Additive exPlanations)**.

Les valeurs SHAP permettent d'identifier les variables qui contribuent à augmenter ou diminuer le risque d'un client.

L'utilisateur peut ainsi consulter non seulement le score obtenu, mais également les principaux facteurs ayant influencé la prédiction.

---

## API FastAPI

L'API constitue l'interface entre le modèle de Machine Learning et l'application Streamlit.

Elle permet notamment :

- de récupérer les clients disponibles ;
- d'obtenir la probabilité de défaut d'un client ;
- d'appliquer le seuil de décision ;
- de récupérer les informations nécessaires à l'interprétation des prédictions.

FastAPI fournit également une documentation interactive de l'API avec **Swagger**  
- En local avec : http://127.0.0.1:8000/docs#/
- Dans le cloud avec l'instance AWS EC2 Exemple http://13.60.210.92:8000/docs#/
  où 13.60.210.92 correspond à l'adresse IPv4 publique de l'instance
    
---

## Interface Streamlit

L'application Streamlit permet à l'utilisateur de sélectionner un client et de visualiser :

- sa probabilité de défaut ;
- la décision associée au score ;
- le seuil de décision ;
- une jauge représentant le niveau de risque par rapport au seuil ;
- l'influence des différentes variables sur la prédiction.

Streamlit communique avec FastAPI par requêtes HTTP et ne réalise pas directement les prédictions.
- En local avec : http://localhost:8501/
- Dans le cloud avec l'instance AWS EC2. Exemple : http://13.60.60.82:8501/
  où 13.60.210.92 correspond à l'adresse IPv4 publique de l'instance.
---

## Tests

Les principales fonctionnalités de l'API sont testées automatiquement avec **Pytest**.

Les tests permettent notamment de vérifier :

  L'API :  
- la récupération de la liste des clients ;
- le fonctionnement de la prédiction ;
- la cohérence dimensionnelle des résultats retournés par l'API ;
- l'API renvoie bien une liste d'identifiants clients, non vide et composée d'entiers ;
- la probabilité entre 0 et 1 ;
- seuil entre 0 et 1 ;
- il existe bien une valeur SHAP et une valeur de variable pour chaque feature.
  
  Le modèle :
- la proba est valide entre 0 et 1 ;
- l'absence de nan après transformation ;
- la récupération des informations nécessaires à l'interprétation SHAP
  
  Le préprocessing :
- la fonction retourne bien un DataFrame ;
- la présence d'une unique ligne par client ;
- l'absence de la cible dans les features

---

## Déploiement

L'application est conteneurisée avec **Docker** et déployée sur une instance **AWS EC2**.

Le processus de déploiement est automatisé avec **GitHub Actions**.

```text
Push sur GitHub
      │
      ▼
GitHub Actions
      │
      ├── Synchronisation du dépôt
      ├── Installation des dépendances
      ├── Exécution des tests Pytest
      ├── Construction de l'image Docker
      └── Redémarrage du conteneur
                    │
                    ▼
                  AWS EC2


```

Cette organisation permet de tester automatiquement l'application avant son redéploiement.

---

## Technologies utilisées

- **Python**
- **Pandas**
- **Scikit-learn**
- **LightGBM**
- **SHAP**
- **FastAPI**
- **Streamlit**
- **Plotly**
- **Pytest**
- **Docker**
- **GitHub Actions**
- **AWS EC2**
