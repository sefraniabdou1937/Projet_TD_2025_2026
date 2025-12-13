import streamlit as st
import os

st.set_page_config(page_title="Télécharger l'Extension", page_icon="🧩")

st.title("🧩 Extension Navigateur")
st.markdown("""
Profitez de la puissance de notre IA directement dans votre navigateur. 
L'extension analyse la page que vous visitez et vous alerte si la revue est suspecte.
""")

# --- SECTION TÉLÉCHARGEMENT ---
file_path = "extension_v1.zip"

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/888/888869.png", width=100)

with col2:
    st.subheader("Version 1.0")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger l'Extension (.zip)",
                data=f,
                file_name="Predatory_Detector_v1.zip",
                mime="application/zip",
                type="primary"
            )
    else:
        st.error("Le fichier 'extension_v1.zip' est introuvable sur le serveur.")

st.divider()

# --- GUIDE D'INSTALLATION ---
st.header("🛠️ Comment l'installer ?")

st.info("💡 Comme c'est une extension privée (non publiée sur le Chrome Store), l'installation est manuelle.")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("#### 1. Décompresser")
    st.write("Téléchargez le fichier ZIP et **extrayez** son contenu dans un dossier.")

with step2:
    st.markdown("#### 2. Mode Développeur")
    st.write("Ouvrez Chrome, allez sur `chrome://extensions` et activez le **Mode développeur** (en haut à droite).")

with step3:
    st.markdown("#### 3. Charger")
    st.write("Cliquez sur **Charger l'extension non empaquetée** et sélectionnez le dossier décompressé.")

st.success("🎉 L'icône de l'extension apparaîtra dans votre barre d'outils !")