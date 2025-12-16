import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib

print("--- 🔬 ANALYSE SCIENTIFIQUE : COURBES D'APPRENTISSAGE ---")

# --- 1. SETUP (Recréation du pipeline identique) ---
# On redéfinit les fonctions pour être sûr
def get_text_data(x): return x['Titre'].astype(str) + " " + x['Publisher'].astype(str)
def get_numeric_data(x): return x[['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']]

# Chargement Données
df = pd.read_csv("data/02_real_world_dataset.csv")
df['Publisher'] = df['Publisher'].fillna('Unknown')
df['Titre'] = df['Titre'].fillna('')
cols_num = ['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']
for col in cols_num: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X = df
y = df['Est_Predateur']

# Reconstruction du Modèle (Exactement le même que V2)
text_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_text_data, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=6000, ngram_range=(1, 3), stop_words='english')),
])
numeric_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_numeric_data, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
preprocessor = FeatureUnion([('nlp', text_pipeline), ('num', numeric_pipeline)])

clf1 = XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=5, min_child_weight=2, gamma=0.1, scale_pos_weight=1.5, random_state=42, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=400, max_depth=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
ensemble_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2)], voting='soft', weights=[2, 1]))
])

# --- 2. CALCUL MATHEMATIQUE DE LA COURBE ---
print("⏳ Calcul des courbes en cours (Validation Croisée sur différents volumes)...")
# On teste avec 20%, 40%, 60%, 80% et 100% des données
train_sizes, train_scores, test_scores = learning_curve(
    ensemble_model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy',
    shuffle=True
)

# Calcul des moyennes et écarts types
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# --- 3. AFFICHAGE DU GRAPHIQUE ---
plt.figure(figsize=(10, 6))
plt.title("Preuve Scientifique : Potentiel d'Amélioration par la Data")
plt.xlabel("Taille du Dataset (Nombre d'échantillons)")
plt.ylabel("Précision (Accuracy)")
plt.grid(True)

# Courbe d'entraînement
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score Entraînement")
# Courbe de validation (Test)
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score Validation (Réalité)")

plt.legend(loc="best")
plt.savefig("learning_curve_proof.png")
plt.show()

# --- 4. INTERPRÉTATION AUTOMATIQUE ---
slope = test_scores_mean[-1] - test_scores_mean[-2]
print("\n--- 📝 CONCLUSION SCIENTIFIQUE ---")
print(f"Dernier score de validation : {test_scores_mean[-1]:.2%}")
if slope > 0.001:
    print("✅ PENTE POSITIVE DETECTÉE : La courbe verte continue de monter.")
    print("   PREUVE : Le modèle n'a pas atteint son plafond.")
    print("   SOLUTION : Ajouter plus de données augmentera mécaniquement le score.")
else:
    print("⚠️ PLATEAU DÉTECTÉ : La courbe stagne.")
    print("   SOLUTION : Il faut complexifier le modèle (plus de features), la data seule ne suffira pas.")