

---

```markdown
# 🛡️ Predatory Journals Detector
> **Évaluation de la Crédibilité Scientifique par Intelligence Artificielle**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Hybrid_Voting-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)

Ce projet propose une solution complète (**End-to-End**) conçue pour assister les chercheurs dans l'identification des revues et éditeurs prédateurs. Il combine une architecture de données robuste, de l'intelligence artificielle hybride et des outils de détection en temps réel.

---

## 🚀 Fonctionnalités Clés

* **🔍 Collecte Automatisée** : Agrégation intelligente de listes noires (Stop-Predatory, Hijacked Journals) et de listes blanches certifiées (DOAJ).
* **🌐 Enrichissement Metadata** : Utilisation asynchrone des API **OpenAlex** et **Crossref** pour extraire des métriques d'impact réelles (citations, DOI, volume de publications).
* **🤖 Modèle Hybride** : Système de classification haute performance basé sur un mécanisme de vote entre **XGBoost** et **Random Forest**.
* **📊 Dashboard Streamlit** : Interface interactive permettant d'analyser une revue et de contribuer à la base de connaissances via l'apprentissage actif (*Active Learning*).
* **🧩 Extension Navigateur** : Analyse instantanée de la crédibilité directement lors de la navigation sur les sites de revues académiques.

---

## 📂 Architecture du Projet

Le dépôt est organisé de la manière suivante pour garantir une modularité maximale :

* `1_collect_data_unified.py` : Pipeline de collecte et de nettoyage des noms de revues.
* `2_enrich_data_async.py` : Script d'enrichissement asynchrone pour la gestion de la latence API.
* `3_train_model_hybrid.py` : Entraînement, optimisation du seuil de décision et sauvegarde du modèle hybride.
* `app.py` : Point d'entrée de l'application interactive Streamlit.
* `api.py` : Backend FastAPI servant l'extension web pour les prédictions en temps réel.
* `/extension` : Code source de l'extension navigateur (Manifest V3).
* `journal_utils.py` : Fonctions utilitaires partagées pour le traitement des données.

---

## 🛠️ Installation et Déploiement

### 1. Configuration de l'environnement
Assurez-vous d'avoir Python 3.9+ installé. Installez ensuite les dépendances :
```bash
pip install -r requirements.txt

```

###2. Lancement du Dashboard (Interface Utilisateur)```bash
streamlit run app.py

```

###3. Activation de l'API (Backend Extension)L'extension nécessite que le serveur API soit actif pour effectuer des analyses :

```bash
python api.py

```

---

##🧩 Extension Navigateur (Contribution Technique)L'extension permet une détection proactive sans interrompre votre flux de travail scientifique.

###Installation Rapide1. Accédez à `chrome://extensions/` dans votre navigateur.
2. Activez le **Mode développeur** (interrupteur en haut à droite).
3. Cliquez sur **Charger l'extension non empaquetée**.
4. Sélectionnez le dossier `/extension` à la racine de ce projet.

###Fonctionnement Technique* **Extraction** : Récupère automatiquement le nom de la revue de l'onglet actif.
* **Inférence** : Transmet les données à `api.py` pour une analyse immédiate via le modèle hybride.
* **Visualisation** : Affiche un badge de couleur (Vert/Rouge) selon le score de risque détecté.

---

##📊 Performance du SystèmeLe modèle est optimisé pour la sécurité maximale des chercheurs, avec un seuil de décision ajusté pour minimiser les risques.

* **Précision Globale** : ~85.83%.
* **Précision (Classe Prédatrice)** : 87%.
* **Seuil de Décision** : 0.56 (optimisé pour le F1-Score).

---

##👥 Équipe du Projet* **Réalisé par** : Abderrahmane Sefrani, Tiab Zayd, et Hanan Gharibi.
* **Cadre** : Module C - Transformation Digitale 2025/2026 - **ENSAH**.
* **Sous l'encadrement de** : Pr. Sara OUALD CHAIB.

---

```

---

### Pourquoi cette version est "Top Pro" ?
1.  **Badges de statut** : En haut du fichier, ils donnent immédiatement les infos techniques (Python, Streamlit).
2.  **Mise en forme hiérarchique** : Utilisation de titres, de lignes horizontales et de citations pour une lecture fluide.
3.  **Blocs de code propres** : Les commandes d'installation sont prêtes à l'emploi.
4.  **Structure GitHub standard** : Elle suit les meilleures pratiques des dépôts open-source majeurs.

Serait-il utile d'ajouter une section montrant des exemples de requêtes JSON pour l'API ? Seriez-vous intéressé par l'ajout d'une section détaillant les étapes de prétraitement NLP (TF-IDF) ?

```