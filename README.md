Predatory Journals Detector : Évaluation de la Crédibilité Scientifique par IA
Ce projet est une solution complète (End-to-End) conçue pour aider les chercheurs à identifier les revues et éditeurs prédateurs grâce à l'intelligence artificielle hybride et à l'enrichissement de données en temps réel.

🚀 Fonctionnalités Principales
Collecte Automatisée : Agrégation de listes noires (Stop-Predatory, Hijacked Journals) et de listes blanches (DOAJ).

Enrichissement Metadata : Utilisation asynchrone des API OpenAlex et Crossref pour récupérer des métriques d'impact (citations, DOI, nombre de travaux).

Modèle Hybride : Classification robuste utilisant un mécanisme de vote entre XGBoost et Random Forest.

Dashboard Streamlit : Interface interactive pour analyser une revue et contribuer à la base de connaissances via l'apprentissage actif (Active Learning).

Extension Navigateur : Analyse instantanée de la crédibilité directement lors de la navigation sur le site d'une revue.

📂 Structure du Projet
Le dépôt est organisé de la manière suivante :

1_collect_data_unified.py : Script de collecte initiale et de nettoyage des noms de revues.

2_enrich_data_async.py : Script d'enrichissement asynchrone via APIs avec gestion de la latence.

3_train_model_hybrid.py : Entraînement, optimisation du seuil et sauvegarde du modèle hybride.

app.py : Point d'entrée de l'application principale Streamlit.

api.py : Backend FastAPI servant l'extension web pour les prédictions en temps réel.

/extension : Code source de l'extension navigateur au format Manifest V3.

journal_utils.py : Fonctions utilitaires partagées pour le traitement des données.

🛠️ Installation et Utilisation
1. Prérequis
Assurez-vous d'avoir Python 3.9+ installé. Installez ensuite les dépendances nécessaires :

Bash

pip install -r requirements.txt
2. Lancer le Dashboard (Interface Utilisateur)
Exécutez la commande suivante pour lancer l'interface Streamlit :

Bash

streamlit run app.py
3. Lancer l'API pour l'Extension Web
L'extension nécessite que le serveur API soit actif pour effectuer des prédictions :

Bash

python api.py
🧩 Extension Navigateur (Contribution Technique)
L'extension permet une détection proactive sans quitter votre flux de travail de recherche.

Installation
Ouvrez Chrome et accédez à chrome://extensions/.

Activez le Mode développeur (interrupteur en haut à droite).

Cliquez sur Charger l'extension non empaquetée.

Sélectionnez le dossier /extension situé à la racine du projet.

Fonctionnement
Analyse au clic : Cliquez sur l'icône de l'extension lorsqu'une page de revue est ouverte.

Communication API : L'extension extrait le nom de la revue depuis l'onglet actif et l'envoie à api.py, qui interroge le modèle et renvoie un score de risque.

Indicateurs visuels : Un badge de couleur (Vert pour fiable / Rouge pour suspect) s'affiche selon le niveau de danger détecté.

📊 Performance du Modèle
Le modèle a été rigoureusement évalué pour garantir la sécurité des utilisateurs :

Accuracy : ~85.83%.

Précision (Classe Prédatrice) : 87%.

Seuil de décision optimal : 0.56 (optimisé pour minimiser les faux négatifs et protéger les chercheurs).

Projet réalisé par Abderrahmane Sefrani, Tiab Zayd, et Hanan Gharibi dans le cadre du Module C - Transformation Digitale 2025/2026 - ENSAH sous l'encadrement de la Pr. Sara OUALD CHAIB.