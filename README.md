# OCR-Impl-mentez-un-mod-le-de-scoring  
Construire un modèle de scoring - Analyser les features - Mettre en production - MLOps - API  
  
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)   
  
Dans le notebook d’entraînement des modèles, générer à l’aide de MLFlow un tracking d'expérimentations
Lancer l’interface web 'UI MLFlow" d'affichage des résultats du tracking  
Réaliser avec MLFlow un stockage centralisé des modèles dans un “model registry”  
Tester le serving MLFlow  
Gérer le code avec le logiciel de version Git  
Partager le code sur Github pour assurer une intégration continue  
Utiliser Github Actions pour le déploiement continu et automatisé du code de l’API sur le cloud  
Concevoir des tests unitaires avec Pytest (ou Unittest) et les exécuter de manière automatisée lors du build réalisé par Github Actions  

Utilisation de la librairie evidently pour détecter dans le futur du Data Drift en production  
Avec l'hypothèse que le dataset “application_train” représente les datas pour la modélisation et le dataset “application_test” représente les datas de nouveaux clients une fois le modèle en production.    
L’analyse à l’aide d’evidently permettra de détecter éventuellement du Data Drift sur les principales features, entre les datas d’entraînement et les datas de production, au travers du tableau HTML d’analyse généré grâce à evidently.




# Projet 7 — Implémentez un modèle de scoring

## Présentation du projet

Ce projet a pour objectif de développer un **modèle de scoring de crédit** permettant d'estimer la probabilité qu'un client rencontre des difficultés de remboursement.

Il s'appuie sur les données du challenge **Home Credit Default Risk**.

Le projet couvre l'ensemble de la chaîne de développement d'un modèle de Machine Learning : préparation des données, modélisation, interprétation des prédictions, création d'une API, développement d'une interface utilisateur et déploiement dans le cloud.

---

## Objectifs

Les principales étapes du projet sont :

- explorer et préparer les données clients ;
- construire les variables utiles à la modélisation ;
- entraîner et comparer plusieurs modèles de classification ;
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

## Architecture de l'application

L'application repose sur une séparation entre l'interface utilisateur et le moteur de scoring.

```text
Utilisateur
    │
    ▼
Streamlit
api_streamlit.py
    │
    │ Requêtes HTTP
    ▼
FastAPI
api_fastapi.py
    │
    ▼
Pipeline de Machine Learning
    │
    ├── Prétraitement
    ├── Modèle LightGBM
    └── Seuil de décision
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
│   └── ...                     # Exploration, préparation et modélisation
│
├── pytests/
│   └── ...                     # Tests automatisés de l'API
│
├── requirements.txt            # Dépendances Python
│
└── README.md                   # Présentation du projet
```

---

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

## Interprétabilité avec SHAP

Afin de rendre les prédictions du modèle plus compréhensibles, l'application utilise **SHAP (SHapley Additive exPlanations)**.

Les valeurs SHAP permettent d'identifier les variables qui contribuent à augmenter ou diminuer le risque prédit pour un client.

L'utilisateur peut ainsi consulter non seulement le score obtenu, mais également les principaux facteurs ayant influencé la prédiction.

---

## API FastAPI

L'API constitue l'interface entre le modèle de Machine Learning et l'application Streamlit.

Elle permet notamment :

- de récupérer les clients disponibles ;
- d'obtenir la probabilité de défaut d'un client ;
- d'appliquer le seuil de décision ;
- de récupérer les informations nécessaires à l'interprétation des prédictions.

FastAPI fournit également une documentation interactive de l'API avec **Swagger**.

---

## Interface Streamlit

L'application Streamlit permet à l'utilisateur de sélectionner un client et de visualiser notamment :

- sa probabilité de défaut ;
- la décision associée au score ;
- le seuil de décision ;
- une jauge représentant le niveau de risque ;
- l'influence des différentes variables sur la prédiction.

Streamlit communique avec FastAPI par requêtes HTTP et ne réalise pas directement les prédictions.

---

## Tests

Les principales fonctionnalités de l'API sont testées automatiquement avec **Pytest**.

Les tests permettent notamment de vérifier :

- la récupération de la liste des clients ;
- le fonctionnement de la prédiction ;
- la cohérence des résultats retournés par l'API ;
- la récupération des informations nécessaires à l'interprétation SHAP.

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
