from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from journal_utils import get_text_data, get_numeric_data
# On importe la fonction de collecte depuis votre fichier app.py
from app import get_live_metrics 

app = FastAPI()

# --- CONFIGURATION CORS (OBLIGATOIRE POUR L'EXTENSION) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les extensions à parler à l'API
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)
# ---------------------------------------------------------

# Chargement du modèle au démarrage
# Assurez-vous que le fichier .pkl est bien dans le même dossier
model = joblib.load("model_xgboost_publisher.pkl")

@app.get("/predict")
def predict_journal(name: str):
    """
    Reçoit un nom de journal, lance l'enquête et retourne le risque.
    """
    # 1. Enquête en direct (OpenAlex/Crossref)
    # On utilise la fonction puissante que vous avez déjà créée dans app.py
    raw_data = get_live_metrics(name)
    
    # 2. Préparation pour le modèle
    df_input = pd.DataFrame([raw_data])
    
    # 3. Prédiction par l'IA
    proba = model.predict_proba(df_input)[0][1]
    
    # 4. Réponse JSON formatée pour l'extension
    return {
        "journal": name,
        "is_predatory": bool(proba > 0.5), # Vrai si le risque > 50%
        "risk_score": round(proba * 100, 1),
        "details": raw_data
    }

# Pour lancer ce serveur, ouvrez un terminal et tapez :
# uvicorn api:app --reload