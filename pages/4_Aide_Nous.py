import streamlit as st
import pandas as pd
import requests
import os
from urllib.parse import quote

# --- CONFIGURATION ---
st.set_page_config(page_title="Contribuer au Dataset", page_icon="🤝", layout="centered")
DATASET_FILE = "data/02_real_world_dataset.csv"
EMAIL = "etudiant@ensah.ma" # Pour l'API Crossref

# --- FONCTION DE RÉCUPÉRATION (Vérification Numérique) ---
def get_live_metrics(journal_name):
    """Récupère les preuves numériques (Citations, DOI...) en temps réel"""
    
    # 1. OpenAlex (Impact & Volume)
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
        url_cr = f"https://api.crossref.org/works?query.container-title={quote(journal_name)}&rows=1&mailto={EMAIL}"
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
    
    # Calcul du Ratio
    works = full_data['oa_works']
    cited = full_data['oa_cited']
    # Évite la division par zéro
    full_data['Impact_Ratio'] = cited / (works + 1) if works >= 0 else 0
    
    return full_data

# --- FONCTION DE SAUVEGARDE ---
def save_entry(data, label):
    """Ajoute la ligne au fichier CSV"""
    
    # Préparation de la ligne (Respect strict des colonnes du dataset)
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

    try:
        if os.path.exists(DATASET_FILE):
            # Vérification doublon
            df_existing = pd.read_csv(DATASET_FILE, usecols=['Titre'])
            if data['Titre'] in df_existing['Titre'].values:
                return "DUPLICATE"
            
            # Ajout (Append)
            df_new.to_csv(DATASET_FILE, mode='a', header=False, index=False)
        else:
            # Création
            df_new.to_csv(DATASET_FILE, mode='w', header=True, index=False)
        return "SUCCESS"
    except Exception as e:
        return str(e)

# --- INTERFACE UTILISATEUR ---
st.title("🤝 Aidez-nous à améliorer l'IA")
st.markdown("""
Cette page vous permet d'ajouter manuellement une revue à la base de données.
Le système va automatiquement effectuer une **vérification numérique** (récupérer les citations et l'éditeur) avant d'enregistrer votre label.
""")

st.divider()

with st.form("add_journal_form"):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        name_input = st.text_input("Nom exact de la revue", placeholder="Ex: Moroccan Journal of...")
    
    with col2:
        label_input = st.selectbox(
            "Classe", 
            options=["Choisir...", "Fiable (0)", "Prédatrice (1)"],
            help="Sélectionnez 'Fiable' si c'est une revue légitime, 'Prédatrice' sinon."
        )
    
    submit_btn = st.form_submit_button("🔍 Vérifier & Ajouter", type="primary")

if submit_btn:
    if not name_input or label_input == "Choisir...":
        st.warning("Veuillez remplir le nom et choisir une classe.")
    else:
        # 1. Conversion du label en chiffre
        label_val = 0 if "Fiable" in label_input else 1
        
        with st.spinner(f"Enquête numérique sur '{name_input}'..."):
            # 2. Récupération des données (Enrichissement)
            metrics = get_live_metrics(name_input)
            
            # 3. Affichage des preuves trouvées (Feedback)
            st.success("Données numériques récupérées !")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Éditeur", metrics['Publisher'])
            c2.metric("Articles", metrics['oa_works'])
            c3.metric("Citations", metrics['oa_cited'])
            c4.metric("Ratio", f"{metrics['Impact_Ratio']:.2f}")
            
            if metrics['Publisher'] == 'Unknown':
                st.caption("⚠️ Attention : Aucun éditeur officiel détecté.")
            
            # 4. Sauvegarde
            result = save_entry(metrics, label_val)
            
            if result == "SUCCESS":
                st.toast("✅ Revue ajoutée avec succès à la base d'entraînement !", icon="💾")
                st.balloons()
                st.info(f"La revue a été enregistrée comme **{label_input}**. Elle sera intégrée au prochain ré-entraînement du modèle.")
            elif result == "DUPLICATE":
                st.error("Cette revue existe déjà dans la base de données.")
            else:
                st.error(f"Erreur lors de la sauvegarde : {result}")

st.divider()
st.caption("Les données ajoutées ici enrichissent le fichier `data/02_real_world_dataset.csv`.")