import streamlit as st  # N'oubliez pas cette ligne au début !

with open("extension_v1.zip", "rb") as f:
    st.download_button(  # <--- C'est ici que c'était écrit "set"
        label="📥 Télécharger l'Extension Chrome",
        data=f,
        file_name="predatory_detector_extension.zip",
        mime="application/zip",
        help="Installez notre IA directement dans votre navigateur !"
    )