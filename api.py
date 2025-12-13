from fastapi import FastAPI
import joblib
import pandas as pd
from journal_utils import get_text_data, get_numeric_data
# On réutilise votre fonction de collecte (copiez-la depuis app.py ou importez-la)
from app import get_live_metrics 

app = FastAPI()

# Chargement du modèle au démarrage
model = joblib.load("model_xgboost_publisher.pkl")

@app.get("/predict")
def predict_journal(name: str):
    # 1. Enquête en direct (OpenAlex/Crossref)
    raw_data = get_live_metrics(name)
    
    # 2. Préparation pour le modèle
    df_input = pd.DataFrame([raw_data])
    
    # 3. Prédiction
    proba = model.predict_proba(df_input)[0][1]
    
    # 4. Réponse JSON
    return {
        "journal": name,
        "is_predatory": bool(proba > 0.5),
        "risk_score": round(proba * 100, 1),
        "details": raw_data
    }

# Pour lancer : uvicorn api:app --reload