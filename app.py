import streamlit as st
import pandas as pd
import joblib
import requests
import os
from urllib.parse import quote
from journal_utils import get_text_data, get_numeric_data  # On importe vos fonctions

# --- CONFIGURATION ---
st.set_page_config(page_title="Predatory AI", page_icon="🛡️", layout="centered")
MODEL_FILE = "model_xgboost_publisher.pkl"

# --- FONCTIONS D'ENQUÊTE EN DIRECT (Mêmes APIs que l'enrichissement) ---
def get_live_metrics(journal_name):
    """Interroge OpenAlex et Crossref en temps réel"""
    
    # 1. OpenAlex (Impact)
    oa_data = {'oa_works': 0, 'oa_cited': 0, 'oa_found': 0}
    try:
        url_oa = f"https://api.openalex.org/sources?search={quote(journal_name)}"
        resp = requests.get(url_oa, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                top = results[0]
                oa_data = {
                    'oa_works': top.get('works_count', 0),
                    'oa_cited': top.get('cited_by_count', 0),
                    'oa_found': 1
                }
    except:
        pass

    # 2. Crossref (Éditeur & DOI)
    cr_data = {'cr_has_doi': 0, 'Publisher': 'Unknown'}
    try:
        url_cr = f"https://api.crossref.org/works?query.container-title={quote(journal_name)}&rows=1"
        resp = requests.get(url_cr, timeout=5)
        if resp.status_code == 200:
            items = resp.json().get('message', {}).get('items', [])
            if items:
                cr_data = {
                    'cr_has_doi': 1,
                    'Publisher': items[0].get('publisher', 'Unknown')
                }
    except:
        pass

    # Fusion
    full_data = {**oa_data, **cr_data}
    full_data['Titre'] = journal_name
    
    # Calcul Ratio
    works = full_data['oa_works']
    cited = full_data['oa_cited']
    full_data['Impact_Ratio'] = cited / (works + 1)
    
    return full_data

# --- INTERFACE ---
st.title("🛡️ Détecteur de Revues Prédatrices")
st.caption("Propulsé par XGBoost • ENSAH Data Project")

# Chargement du modèle
if not os.path.exists(MODEL_FILE):
    st.error("❌ Modèle introuvable. Avez-vous lancé l'étape 3 ?")
    st.stop()

model = joblib.load(MODEL_FILE)

# Champ de recherche
query = st.text_input("Entrez le nom de la revue à vérifier :", placeholder="Ex: International Journal of Science...")

if st.button("Lancer l'Audit", type="primary"):
    if not query:
        st.warning("Veuillez entrer un nom.")
    else:
        with st.spinner("🕵️ L'IA enquête sur les bases de données mondiales..."):
            # 1. Récupération des données live
            raw_data = get_live_metrics(query)
            
            # 2. Création DataFrame (Format identique à l'entraînement)
            df_input = pd.DataFrame([raw_data])
            
            # 3. Prédiction
            proba = model.predict_proba(df_input)[0][1] # Probabilité d'être Arnaque (Classe 1)
            percent = round(proba * 100, 1)
            
            # 4. Affichage Résultat
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Jauge de risque
                if percent < 40:
                    st.image("https://img.icons8.com/color/96/guarantee.png", width=80)
                    st.success(f"**FIABLE**\n\nRisque : {percent}%")
                elif percent > 70:
                    st.image("https://img.icons8.com/color/96/skull.png", width=80)
                    st.error(f"**DANGER**\n\nRisque : {percent}%")
                else:
                    st.image("https://img.icons8.com/color/96/high-priority.png", width=80)
                    st.warning(f"**SUSPECT**\n\nRisque : {percent}%")

            with col2:
                st.subheader("Rapport d'Enquête")
                st.write(f"**Éditeur détecté :** {raw_data['Publisher']}")
                st.write(f"**Impact (Citations) :** {raw_data['oa_cited']}")
                st.write(f"**Volume (Articles) :** {raw_data['oa_works']}")
                
                # Explication simple
                if raw_data['Publisher'] == 'Unknown':
                    st.markdown("🔴 **Alerte :** Aucun éditeur officiel trouvé.")
                if raw_data['oa_cited'] == 0 and raw_data['oa_works'] > 0:
                    st.markdown("🔴 **Alerte :** La revue publie mais personne ne la cite (Impact nul).")
                if "International" in query and percent > 50:
                    st.markdown("🟠 **Note :** Usage suspect du mot 'International'.")

            # 5. Affichage des données brutes (Amélioré)
            with st.expander("Voir les données techniques"):
                # On crée une copie pour l'affichage
                df_display = df_input.copy()
                
                # Renommage des colonnes pour être compréhensible
                rename_dict = {
                    'Titre': 'Nom de la Revue',
                    'Publisher': 'Éditeur',
                    'oa_works': 'Nb Articles',
                    'oa_cited': 'Nb Citations',
                    'oa_found': 'Trouvé OpenAlex',
                    'cr_has_doi': 'DOI Valide',
                    'Impact_Ratio': 'Ratio Impact'
                }
                df_display = df_display.rename(columns=rename_dict)
                
                # Remplacement des 0/1 par Non/Oui
                if 'Trouvé OpenAlex' in df_display.columns:
                    df_display['Trouvé OpenAlex'] = df_display['Trouvé OpenAlex'].map({0: 'Non', 1: 'Oui'})
                if 'DOI Valide' in df_display.columns:
                    df_display['DOI Valide'] = df_display['DOI Valide'].map({0: 'Non', 1: 'Oui'})
                
                # Affichage
                st.dataframe(df_display, use_container_width=True)