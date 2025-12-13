import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_FILE = "data/02_real_world_dataset.csv"
MODEL_FILE = "model_xgboost_publisher.pkl"

print("--- 🧠 ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE HYBRIDE (NLP + STATS) ---")

# 1. Chargement et Nettoyage
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Dataset chargé : {len(df)} lignes.")
except FileNotFoundError:
    print("❌ Erreur : Fichier introuvable. Lancez l'étape 2 d'abord.")
    exit()

# On remplit les vides pour le texte par "Unknown" et les chiffres par -1 ou 0
df['Publisher'].fillna('Unknown', inplace=True)
df['Titre'].fillna('', inplace=True)
df.fillna(0, inplace=True)

# 2. Définition des Fonctions de Transformation (Les "Yeux" du modèle)
# Ces fonctions doivent être accessibles lors de l'utilisation du modèle (App)

def get_text_data(x):
    # On combine Titre et Éditeur pour donner un maximum de contexte sémantique
    return x['Titre'].astype(str) + " " + x['Publisher'].astype(str)

def get_numeric_data(x):
    # On prend les preuves factuelles
    return x[['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']]

# Sauvegarde des fonctions utilitaires pour l'application Streamlit
utils_code = """
import pandas as pd
def get_text_data(x):
    return x['Titre'].astype(str) + " " + x['Publisher'].astype(str)
def get_numeric_data(x):
    return x[['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']]
"""
with open("journal_utils.py", "w") as f:
    f.write(utils_code)
print("📦 Fonctions utilitaires sauvegardées dans 'journal_utils.py'")

# 3. Construction du Pipeline Hybride

# BRANCHE 1 : Traitement du Texte (NLP)
# Transforme les mots "Oxford", "Journal", "Science" en vecteurs mathématiques
text_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_text_data, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')),
])

# BRANCHE 2 : Traitement des Chiffres (Stats)
# Normalise les citations (car 10000 citations ne doit pas écraser le reste)
numeric_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_numeric_data, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # Sécurité
    ('scaler', StandardScaler()) # Met tout à la même échelle
])

# FUSION : On met tout ensemble
preprocessor = FeatureUnion([
    ('nlp_features', text_pipeline),
    ('num_features', numeric_pipeline)
])

# LE CERVEAU : XGBoost
# Paramètres optimisés pour éviter le sur-apprentissage
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(
        n_estimators=300,       # Nombre d'arbres
        learning_rate=0.05,     # Vitesse d'apprentissage (douce)
        max_depth=8,            # Profondeur des arbres (complexité)
        subsample=0.8,          # Évite de trop apprendre par coeur
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=-1               # Utilise tous les coeurs du PC
    ))
])

# 4. Préparation de l'entraînement
X = df
y = df['Est_Predateur']

# Séparation Train (80%) / Test (20%) avec stratification (garde la proportion arnaque/fiable)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n🚀 Entraînement sur {len(X_train)} revues...")
model.fit(X_train, y_train)

# 5. Évaluation des Performances
print("\n--- 📊 RÉSULTATS DU CRASH TEST ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Précision Globale (Accuracy) : {accuracy:.2%}")

print("\n🔍 Rapport Détaillé :")
print(classification_report(y_test, y_pred, target_names=['Fiable (0)', 'Prédateur (1)']))

# Matrice de confusion simplifiée
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"✅ Vrais Fiables détectés : {tn}")
print(f"🕵️ Vrais Prédateurs détectés : {tp}")
print(f"⚠️ Fausses Alertes (Fiables classés Arnaque) : {fp}")
print(f"☠️ Arnaques Ratées (Arnaques classées Fiables) : {fn}")

# 6. Sauvegarde
joblib.dump(model, MODEL_FILE)
print(f"\n💾 Modèle sauvegardé : {MODEL_FILE}")
print("👉 Vous pouvez maintenant lancer l'application Streamlit !")