**🛡️ Predatory Journals Detector** Évaluation de la Crédibilité Scientifique par Intelligence Artificielle*Ce projet propose une solution complète (**End-to-End**) conçue pour assister les chercheurs dans l'identification des revues et éditeurs prédateurs. Il combine une architecture de données robuste, de l'intelligence artificielle hybride et des outils de détection en temps réel.

---

##🚀 Fonctionnalités Principales* **Collecte Automatisée** : Agrégation intelligente de listes noires (Stop-Predatory, Hijacked Journals) et de listes blanches certifiées (DOAJ).
* **Enrichissement Metadata** : Utilisation asynchrone des API **OpenAlex** et **Crossref** pour extraire des métriques d'impact réelles (citations, DOI, volume de publications).
* **Modèle Hybride** : Système de classification haute performance basé sur un mécanisme de vote entre **XGBoost** et **Random Forest**.
* **Dashboard Streamlit** : Interface interactive permettant d'analyser une revue et de contribuer à la base de connaissances via l'apprentissage actif (*Active Learning*).
* **Extension Navigateur** : Analyse instantanée de la crédibilité directement lors de la navigation sur les sites de revues académiques.

---

##📂 Structure du Projet* `1_collect_data_unified.py` : Pipeline de collecte et de nettoyage des noms de revues.
* `2_enrich_data_async.py` : Script d'enrichissement asynchrone pour la gestion de la latence API.
* `3_train_model_hybrid.py` : Entraînement, optimisation du seuil de décision et sauvegarde du modèle hybride.
* `app.py` : Point d'entrée de l'application interactive Streamlit.
* `api.py` : Backend FastAPI servant l'extension web pour les prédictions en temps réel.
* `/extension` : Code source de l'extension navigateur (Manifest V3).
* `journal_utils.py` : Fonctions utilitaires partagées pour le traitement des données.

---

##🛠️ Installation et Utilisation###1. PrérequisAssurez-vous d'avoir Python 3.9+ d'installé. Installez ensuite les dépendances :

```bash
pip install -r requirements.txt

```

###2. Lancer le Dashboard (Interface Utilisateur)```bash
streamlit run app.py

```

###3. Lancer l'API pour l'ExtensionL'extension nécessite que le serveur API soit actif pour effectuer des analyses :

```bash
python api.py

```

---

##🧩 Extension Navigateur (Contribution Technique)L'extension permet une détection proactive sans interrompre votre flux de travail scientifique.

###Installation1. Ouvrez Chrome et accédez à `chrome://extensions/`.
2. Activez le **Mode développeur** (interrupteur en haut à droite).
3. Cliquez sur **Charger l'extension non empaquetée**.
4. Sélectionnez le dossier `/extension` à la racine de ce projet.

###Fonctionnement* **Analyse au clic** : Extrait automatiquement le nom de la revue de l'onglet actif.
* **Communication API** : Transmet les données à `api.py` pour une inférence immédiate.
* **Indicateurs visuels** : Affiche un badge de couleur (Vert/Rouge) selon le score de risque détecté.

---

##📊 Performance du ModèleLe modèle est optimisé pour la sécurité maximale des chercheurs :

* **Précision Globale** : ~85.83%.
* **Précision (Classe Prédatrice)** : 87%.
* **Seuil de Décision** : 0.56 (ajusté pour minimiser les faux négatifs).

---

> **Note de l'équipe** : Projet réalisé par **Abderrahmane Sefrani**, **Tiab Zayd**, et **Hanan Gharibi** dans le cadre du *Module C - Transformation Digitale 2025/2026* à l'**ENSAH**, sous l'encadrement de la **Pr. Sara OUALD CHAIB**.