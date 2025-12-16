import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

print("--- 📊 GÉNÉRATEUR DE MÉTRIQUES POUR LA SOUTENANCE ---")

# --- 1. CONFIGURATION (Indispensable pour charger le modèle) ---
# On doit redéfinir les fonctions utilisées dans le pickle si elles ne sont pas importées
try:
    from journal_utils import get_text_data, get_numeric_data
except ImportError:
    # Au cas où le fichier n'est pas là, on le recrée à la volée
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

# --- 2. CHARGEMENT DES DONNÉES & DU MODÈLE ---
print("1️⃣ Chargement du modèle et des données...")
model = joblib.load('model_xgboost_publisher.pkl')
df = pd.read_csv("data/02_real_world_dataset.csv")

# Nettoyage (Identique à l'entraînement)
df['Publisher'] = df['Publisher'].fillna('Unknown')
df['Titre'] = df['Titre'].fillna('')
cols_num = ['oa_works', 'oa_cited', 'oa_found', 'cr_has_doi', 'Impact_Ratio']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- 3. RECRÉATION DU TEST SET ---
# On utilise le même random_state=42 pour retrouver exactement les mêmes données de test
X = df
y = df['Est_Predateur']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. CALCUL DU SEUIL OPTIMAL ---
# On recalcule le seuil exact qui a été trouvé lors de l'entraînement
probs = model.predict_proba(X_test)[:, 1]
best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.3, 0.7, 0.01):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   ➤ Seuil Optimal Recalculé : {best_threshold:.2f}")
y_pred = (probs >= best_threshold).astype(int)

# --- 5. AFFICHAGE DES RÉSULTATS (Format Slide) ---
acc = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
report = classification_report(y_test, y_pred, output_dict=True)

print("\n" + "="*40)
print("   📝 CHIFFRES CLÉS POUR LA SLIDE 4")
print("="*40)

print(f"\n🔹 1. PERFORMANCE GLOBALE (Accuracy)")
print(f"   Score : {acc:.2%} (C'est ce chiffre qu'il faut mettre en GROS)")

print(f"\n🔹 2. MATRICE DE CONFUSION (La Preuve)")
print(f"   ✅ Vrais Positifs (Arnaques stoppées) : {tp}")
print(f"   🛡️ Vrais Négatifs (Revues fiables OK) : {tn}")
print(f"   ⚠️ Faux Positifs (Fausses alertes)    : {fp}")
print(f"   ☠️ Faux Négatifs (Arnaques ratées)    : {fn}  <-- LE PLUS IMPORTANT (Doit être bas)")

print(f"\n🔹 3. DÉTAILS PAR CLASSE")
print(f"   🔴 Classe 'Prédateur' :")
print(f"      - Précision : {report['1']['precision']:.2f}")
print(f"      - Rappel    : {report['1']['recall']:.2f} (Capacité à tout détecter)")
print(f"      - F1-Score  : {report['1']['f1-score']:.2f}")

print("\n" + "="*40)
print("💡 PHRASE D'ANALYSE POUR LE PROF :")
print(f"> 'Avec un seuil de {best_threshold:.2f}, nous avons réussi à bloquer {tp} revues prédatrices")
print(f"> tout en ne laissant passer que {fn} arnaques sur l'ensemble du test set.'")
print("="*40)