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
