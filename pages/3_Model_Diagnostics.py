import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Diagnostic Modèle", page_icon="🩺", layout="wide")

# Ajout du dossier parent au path pour importer journal_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from journal_utils import get_text_data, get_numeric_data
except ImportError:
    st.error("❌ Impossible de charger 'journal_utils.py'. Assurez-vous qu'il est à la racine.")
    st.stop()

# Chemins
MODEL_FILE = "model_xgboost_publisher.pkl"
DATA_FILE = "data/02_real_world_dataset.csv"

# --- TITRE ---
st.title("🩺 Diagnostic Complet du Modèle (XGBoost + RF)")
st.markdown("---")

# --- 1. CHARGEMENT ET PRÉPARATION ---
@st.cache_data
def load_and_evaluate():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
        return None, None, None, None

    # Chargement
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    # Nettoyage (Idem entraînement)
    df['Publisher'] = df['Publisher'].fillna('Unknown')
    df['Titre'] = df['Titre'].fillna('')
    cols_num = ['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Split (Même graine random_state=42 pour reproduire les résultats)
    X = df
    y = df['Est_Predateur']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probabilités pour le test (pour l'analyse fine)
    probs_test = model.predict_proba(X_test)[:, 1]

    return df, y_train, y_pred_train, y_test, y_pred_test, probs_test

df, y_train, y_pred_train, y_test, y_pred_test, probs_test = load_and_evaluate()

if df is None:
    st.error("Fichiers manquants (Modèle ou Data). Lancez le pipeline d'abord.")
    st.stop()

# --- 2. ANALYSE OVERFITTING (Train vs Test) ---
st.header("1. Analyse de la Robustesse (Overfitting)")

col1, col2, col3 = st.columns(3)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
gap = acc_train - acc_test

with col1:
    st.metric("🎯 Précision TRAIN (Apprentissage)", f"{acc_train:.2%}")
    st.caption("Capacité du modèle à apprendre par cœur.")

with col2:
    st.metric("🏆 Précision TEST (Réalité)", f"{acc_test:.2%}")
    st.caption("Capacité du modèle à généraliser.")

with col3:
    st.metric("⚠️ Écart (Gap)", f"{gap:.2%}", delta_color="inverse")
    
    if gap < 0.05:
        st.success("✅ **EXCELLENT** : Pas d'overfitting (< 5%)")
    elif gap < 0.10:
        st.warning("⚠️ **ATTENTION** : Léger overfitting (5-10%)")
    else:
        st.error("🚨 **DANGER** : Fort overfitting (> 10%)")

# --- 3. MATRICE DE CONFUSION ---
st.header("2. Matrice de Confusion (Où se trompe-t-il ?)")

col_conf, col_details = st.columns([1, 1])

with col_conf:
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fiable (0)', 'Prédateur (1)'],
                yticklabels=['Fiable (0)', 'Prédateur (1)'])
    plt.xlabel('Prédit par IA')
    plt.ylabel('Réalité')
    plt.title('Matrice de Confusion (Test Set)')
    st.pyplot(fig_cm)

with col_details:
    tn, fp, fn, tp = cm.ravel()
    st.write("#### 🔍 Analyse des Erreurs :")
    
    st.info(f"✅ **Vrais Fiables : {tn}** (Revues honnêtes bien validées)")
    st.success(f"🕵️ **Vrais Prédateurs : {tp}** (Arnaques bien bloquées)")
    
    st.warning(f"⚠️ **Fausses Alertes (FP) : {fp}**\nRevues fiables classées comme arnaques (Paranoïa du modèle).")
    
    st.error(f"☠️ **Arnaques Ratées (FN) : {fn}**\nLe plus dangereux ! Arnaques passées entre les mailles du filet.")
    
    if fn < 70:
        # st.balloons()  <-- LIGNE SUPPRIMÉE
        st.caption("🏆 Objectif de sécurité atteint (< 70 ratés)")

# --- 4. CORRÉLATIONS ---
st.header("3. Corrélations des Données")
st.markdown("Quelles caractéristiques (Features) sont les plus liées à la fraude ?")

# On calcule la corrélation sur tout le dataset
corr_cols = ['Est_Predateur', 'oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']
corr_matrix = df[corr_cols].corr()

fig_corr, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(corr_matrix[['Est_Predateur']].sort_values(by='Est_Predateur', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Corrélation avec la cible 'Est_Predateur'")
st.pyplot(fig_corr)

st.info("""
**Interprétation :**
* Une valeur proche de **1** signifie que si ce chiffre monte, le risque d'arnaque monte.
* Une valeur proche de **-1** signifie que si ce chiffre monte, le risque d'arnaque **baisse** (ex: Impact_Ratio).
""")

# --- 5. RAPPORT D'APPROCHE ---
st.header("4. Architecture Technique")

with st.expander("Voir les détails de l'algorithme utilisé", expanded=True):
    st.markdown("""
    ### 🧠 Architecture Hybride "Voting Classifier"
    
    Ce modèle n'est pas une simple IA, c'est un **comité d'experts** composé de deux algorithmes qui votent ensemble :
    
    1.  **XGBoost (Expert Précision)** 🚀
        * **Rôle :** Analyse les motifs complexes et les relations non-linéaires.
        * **Spécialité :** Détecte les signaux faibles dans le texte (Titre + Éditeur).
        * **Configuration :** Profondeur limitée (Max Depth=5) pour éviter l'apprentissage par cœur.
        * **Sécurité :** Utilise `scale_pos_weight=1.5` pour punir sévèrement les arnaques ratées.
    
    2.  **Random Forest (Expert Stabilité)** 🌳
        * **Rôle :** Assure la robustesse globale en moyennant des centaines d'arbres de décision.
        * **Spécialité :** Très bon pour traiter les métriques brutes (Citations, Ratio).
        * **Poids :** Compte pour 1/3 de la décision finale.
    
    ### ⚙️ Traitement des Données (Pipeline)
    * **NLP (Texte) :** TF-IDF sur 6000 mots-clés (1-3 ngrams). Combine le Titre et l'Éditeur.
    * **Stats (Numérique) :** Standardisation des citations et du ratio d'impact.
    * **Seuil de Décision :** Optimisé dynamiquement (pas juste 50%) pour maximiser le F1-Score.
    """)