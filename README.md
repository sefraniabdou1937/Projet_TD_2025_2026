
---
# 🛡️ Predatory Journals Detector
> **Système d'Évaluation de la Crédibilité Scientifique par Intelligence Artificielle**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Hybrid_Voting_Classifier-green.svg)]()

Ce projet propose une solution **End-to-End** conçue pour assister les chercheurs dans l'identification des revues et éditeurs prédateurs. Il combine une architecture de données robuste, de l'IA hybride et des outils de détection en temps réel via une extension navigateur.

---

## 🚀 Fonctionnalités Clés

* **🔍 Collecte Automatisée** : Agrégation intelligente de listes noires (Stop-Predatory, Hijacked Journals) et de listes blanches certifiées (DOAJ).
* **🌐 Enrichissement Metadata** : Utilisation asynchrone des API **OpenAlex** et **Crossref** pour extraire des métriques d'impact réelles (citations, DOI, volume de publications).
* **🤖 Modèle Hybride** : Système de classification basé sur un mécanisme de vote entre **XGBoost** et **Random Forest**.
* **📊 Dashboard Streamlit** : Interface interactive pour l'analyse de revues et la contribution via l'apprentissage actif (*Active Learning*).
* **🧩 Extension Navigateur** : Analyse instantanée de la crédibilité directement lors de la navigation sur les sites académiques.

---

## 📂 Structure du Projet

* `1_collect_data_unified.py` : Pipeline de collecte et nettoyage des données.
* `2_enrich_data_async.py` : Script d'enrichissement asynchrone via APIs.
* `3_train_model_hybrid.py` : Entraînement et optimisation du modèle ML.
* `app.py` : Point d'entrée de l'application interactive Streamlit.
* `api.py` : Backend FastAPI servant l'extension web.
* `/extension` : Code source de l'extension navigateur (Manifest V3).
* `journal_utils.py` : Fonctions utilitaires partagées.

---

## 🛠️ Installation et Déploiement

### 1. Configuration
Installez les dépendances nécessaires :
```bash
pip install -r requirements.txt

```

###2. Dashboard (Interface Utilisateur)Lancez l'interface de contrôle :

```bash
streamlit run app.py

```

###3. API Backend (Pour l'extension)L'extension nécessite que le serveur API soit actif :

```bash
python api.py

```

---

##🧩 Extension Navigateur (Contribution Technique)L'extension permet une détection proactive directement dans votre flux de travail :

1. Accédez à `chrome://extensions/`.
2. Activez le **Mode développeur**.
3. Cliquez sur **Charger l'extension non empaquetée**.
4. Sélectionnez le dossier `/extension` de ce projet.

**Fonctionnement** : Elle extrait le nom de la revue de l'onglet actif, l'envoie à l'API, et affiche un badge de couleur (Vert/Rouge) selon le score de risque détecté.

---

##📊 Performance du Système* **Précision Globale** : ~85.83%.
* **Précision (Classe Prédatrice)** : 87%.
* **Seuil de Décision** : 0.56 (Optimisé pour minimiser les faux négatifs).

---

##👥 Équipe & Encadrement* **Réalisé par** : Abderrahmane Sefrani, Tiab Zayd, et Hanan Gharibi.
* **Cadre** : Module C - Transformation Digitale 2025/2026 - **ENSAH**.
* **Encadrant** : Pr. Sara OUALD CHAIB.

```

```