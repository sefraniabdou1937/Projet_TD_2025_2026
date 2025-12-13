import streamlit as st
import pandas as pd
import joblib
import requests
import os
from urllib.parse import quote
from journal_utils import get_text_data, get_numeric_data

# --- CONFIGURATION ---
st.set_page_config(page_title="Predatory AI", page_icon="🛡️", layout="centered")
MODEL_FILE = "model_xgboost_publisher.pkl"
DATASET_FILE = "data/02_real_world_dataset.csv"  # Le fichier qui nourrit l'IA

# --- FONCTION DE SAUVEGARDE (AUTO-APPRENTISSAGE) ---
def save_to_knowledge_base(data, label):
    """Ajoute une nouvelle revue à la base de connaissances si elle n'existe pas déjà"""
    try:
        # 1. On prépare la ligne à sauvegarder (Même format que le CSV d'entraînement)
        new_row = {
            'Titre': data['Titre'],
            'Est_Predateur': label,
            'oa_works': data['oa_works'],
            'oa_cited': data['oa_cited'],
            'oa_found': data['oa_found'],
            'cr_has_doi': data['cr_has_doi'],
            'Publisher': data['Publisher'],
            'Impact_Ratio': data['Impact_Ratio']
        }
        
        df_new = pd.DataFrame([new_row])

        # 2. Vérification : Est-ce que le fichier existe ?
        if os.path.exists(DATASET_FILE):
            # On charge juste les titres pour vérifier si ça existe déjà (pour ne pas tout charger)
            existing_titles = pd.read_csv(DATASET_FILE, usecols=['Titre'])['Titre'].tolist()
            
            if data['Titre'] in existing_titles:
                return False # Déjà connu, on ne fait rien
            
            # 3. Ajout à la fin du fichier (mode 'a' = append)
            df_new.to_csv(DATASET_FILE, mode='a', header=False, index=False)
        else:
            # Si le fichier n'existe pas, on le crée
            df_new.to_csv(DATASET_FILE, mode='w', header=True, index=False)
            
        return True
    except Exception as e:
        print(f"Erreur de sauvegarde : {e}")
        return False

# --- FONCTIONS D'ENQUÊTE EN DIRECT ---
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
st.caption("Propulsé par XGBoost • Auto-Apprenant 🧠")

if not os.path.exists(MODEL_FILE):
    st.error("❌ Modèle introuvable.")
    st.stop()

model = joblib.load(MODEL_FILE)

query = st.text_input("Entrez le nom de la revue à vérifier :", placeholder="Ex: International Journal of Science...")

if st.button("Lancer l'Audit", type="primary"):
    if not query:
        st.warning("Veuillez entrer un nom.")
    else:
        with st.spinner("🕵️ L'IA enquête et apprend..."):
            # 1. Données Live
            raw_data = get_live_metrics(query)
            df_input = pd.DataFrame([raw_data])
            
            # 2. Prédiction
            proba = model.predict_proba(df_input)[0][1]
            percent = round(proba * 100, 1)
            
            # 3. Logique d'Auto-Apprentissage (Active Learning)
            saved = False
            label_saved = None
            
            # Si très sûr que c'est une Arnaque (> 70%)
            if percent > 70:
                saved = save_to_knowledge_base(raw_data, 1)
                label_saved = "Arnaque"
                
            # Si très sûr que c'est Fiable (< 40%)
            elif percent < 40:
                saved = save_to_knowledge_base(raw_data, 0)
                label_saved = "Fiable"

            # 4. Affichage
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if percent < 40:
                    st.image("https://img.icons8.com/color/96/guarantee.png", width=80)
                    st.success(f"**FIABLE**\n\nRisque : {percent}%")
                elif percent > 70:
                    st.image("https://img.icons8.com/color/96/skull.png", width=80)
                    st.error(f"**DANGER**\n\nRisque : {percent}%")
                else:
                    st.image("https://img.icons8.com/color/96/high-priority.png", width=80)
                    st.warning(f"**SUSPECT**\n\nRisque : {percent}%")
                
                # Feedback visuel de l'apprentissage
                if saved:
                    st.toast(f"✅ Appris ! Ajouté à la base comme '{label_saved}'.", icon="🧠")

            with col2:
                st.subheader("Rapport d'Enquête")
                st.write(f"**Éditeur :** {raw_data['Publisher']}")
                st.write(f"**Impact :** {raw_data['oa_cited']} citations")
                
                if raw_data['Publisher'] == 'Unknown':
                    st.caption("🔴 Pas d'éditeur officiel.")
                if raw_data['oa_cited'] == 0 and raw_data['oa_works'] > 0:
                    st.caption("🔴 Impact nul malgré des publications.")

            with st.expander("Voir les données techniques"):
                df_display = df_input.copy()
                rename_dict = {
                    'Titre': 'Nom', 'Publisher': 'Éditeur', 'oa_works': 'Nb Articles',
                    'oa_cited': 'Nb Citations', 'oa_found': 'Trouvé OpenAlex',
                    'cr_has_doi': 'DOI Valide', 'Impact_Ratio': 'Ratio Impact'
                }
                df_display = df_display.rename(columns=rename_dict)
                st.dataframe(df_display, use_container_width=True)