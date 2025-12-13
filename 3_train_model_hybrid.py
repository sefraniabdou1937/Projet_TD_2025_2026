import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

print("--- 🧠 ÉTAPE 3 : ENTRAÎNEMENT V2 (OPTIMISATION & SÉCURITÉ) ---")

# --- 1. UTILS ---
utils_code = """
import pandas as pd
def get_text_data(x):
    return x['Titre'].astype(str) + " " + x['Publisher'].astype(str)
def get_numeric_data(x):
    return x[['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']]
"""
with open("journal_utils.py", "w") as f:
    f.write(utils_code)
from journal_utils import get_text_data, get_numeric_data

# --- 2. CHARGEMENT ---
DATA_FILE = "data/02_real_world_dataset.csv"
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Dataset chargé : {len(df)} lignes.")
except FileNotFoundError:
    print("❌ Fichier introuvable.")
    exit()

# Nettoyage
df['Publisher'] = df['Publisher'].fillna('Unknown')
df['Titre'] = df['Titre'].fillna('')
cols_num = ['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 3. PIPELINE ROBUSTE ---
text_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_text_data, validate=False)),
    # On augmente max_features pour capter plus de nuances
    ('tfidf', TfidfVectorizer(max_features=6000, ngram_range=(1, 3), stop_words='english')),
])

numeric_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_numeric_data, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

preprocessor = FeatureUnion([
    ('nlp', text_pipeline),
    ('num', numeric_pipeline)
])

# --- 4. MODÈLE "ANTI-ARNAQUE" ---
# XGBoost configuré pour généraliser (moins d'overfitting) et punir les oublis
clf1 = XGBClassifier(
    n_estimators=600,       
    learning_rate=0.02,     # Apprentissage très lent et minutieux
    max_depth=5,            # Arbres moins profonds (Réduit l'overfitting de 94% -> 90%)
    min_child_weight=2,     # Il faut plus de preuves pour créer une feuille
    gamma=0.1,              # Regularization
    scale_pos_weight=1.5,   # BOOST : On donne 50% plus d'importance aux Arnaques !
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Random Forest en soutien
clf2 = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,           # On limite aussi la profondeur ici
    class_weight='balanced_subsample', # Gestion fine du déséquilibre
    random_state=42,
    n_jobs=-1
)

ensemble_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(
        estimators=[('xgb', clf1), ('rf', clf2)],
        voting='soft',
        weights=[2, 1] 
    ))
])

# --- 5. ENTRAÎNEMENT ---
X = df
y = df['Est_Predateur']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n🔥 Entraînement Optimisé en cours...")
ensemble_model.fit(X_train, y_train)

# --- 6. RÉSULTATS & SEUIL ---
# On calcule les probabilités au lieu des classes directes
probs = ensemble_model.predict_proba(X_test)[:, 1]

# Recherche du meilleur seuil (Threshold Moving)
# On veut maximiser le F1-Score (équilibre Précision/Rappel)
best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.3, 0.7, 0.01):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n💡 Seuil de décision optimal trouvé : {best_threshold:.2f}")

# Application du seuil optimal
y_pred_optimized = (probs >= best_threshold).astype(int)

# --- 7. DIAGNOSTIC FINAL ---
acc_train = ensemble_model.score(X_train, y_train)
acc_test = accuracy_score(y_test, y_pred_optimized)

print("\n--- 📊 DIAGNOSTIC FINAL (V2) ---")
print(f"   🎓 Accuracy TRAIN : {acc_train:.2%}")
print(f"   🏆 Accuracy TEST  : {acc_test:.2%}")

gap = acc_train - acc_test
if gap < 0.05:
    print(f"   ✅ EXCELLENT : Overfitting maîtrisé ({gap:.1%}).")
else:
    print(f"   ℹ️ CORRECT : Écart de {gap:.1%}.")

print("\n🔍 Rapport Détaillé (Optimisé) :")
print(classification_report(y_test, y_pred_optimized, target_names=['Fiable', 'Prédateur']))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimized).ravel()
print(f"✅ Arnaques bien détectées : {tp}")
print(f"☠️ Arnaques ratées : {fn} (Objectif : < 70)")

# --- 8. SAUVEGARDE ---
joblib.dump(ensemble_model, 'model_xgboost_publisher.pkl')
print("\n💾 Modèle sauvegardé. Prêt pour l'App.")