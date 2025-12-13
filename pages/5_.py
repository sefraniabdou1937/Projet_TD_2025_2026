with open("extension_v1.zip", "rb") as f:
    st.download_button(
        label="📥 Télécharger l'Extension Chrome",
        data=f,
        file_name="predatory_detector_extension.zip",
        mime="application/zip",
        help="Installez notre IA directement dans votre navigateur !"
    )