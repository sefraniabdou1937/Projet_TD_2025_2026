import pandas as pd
import requests
import os
import io
import random

# --- CONFIGURATION ---
OUTPUT_FILE = "data/01_journal_names.csv"
DOAJ_CSV_URL = "https://doaj.org/csv"  # Le lien magique pour tout avoir d'un coup
URL_BAD_JOURNALS = "https://raw.githubusercontent.com/stop-predatory-journals/stop-predatory-journals.github.io/master/_data/journals.csv"
URL_BAD_HIJACKED = "https://raw.githubusercontent.com/predatory-journals/hijacked-journals/main/hijacked-journals.csv"
URL_BAD_PUBLISHERS = "https://raw.githubusercontent.com/stop-predatory-journals/stop-predatory-journals.github.io/master/_data/publishers.csv"

if not os.path.exists('data'):
    os.makedirs('data')

print("--- 📑 COLLECTE UNIFIÉE : LE SCRIPT ULTIME ---")

# =========================================================
# 1. RÉCUPÉRATION DES PRÉDATEURS (SOURCES GITHUB)
# =========================================================
def get_predators():
    print("\n1️⃣  Récupération des Revues Prédatrices...")
    bad_names = set()

    # Source A : Liste classique
    try:
        print("   ↳ Stop-Predatory-Journals...")
        df = pd.read_csv(URL_BAD_JOURNALS)
        bad_names.update(df['name'].dropna().unique().tolist())
    except Exception as e:
        print(f"     ❌ Erreur : {e}")

    # Source B : Revues Détournées
    try:
        print("   ↳ Hijacked Journals...")
        try:
            df = pd.read_csv(URL_BAD_HIJACKED, on_bad_lines='skip')
        except:
            df = pd.read_csv(URL_BAD_HIJACKED, error_bad_lines=False)
        
        # Trouver la colonne titre
        col = next((c for c in df.columns if 'Title' in c or 'Journal' in c), None)
        if col:
            bad_names.update(df[col].dropna().unique().tolist())
    except:
        pass

    # Source C : Éditeurs douteux
    try:
        print("   ↳ Éditeurs Prédateurs...")
        df = pd.read_csv(URL_BAD_PUBLISHERS)
        bad_names.update(df['name'].dropna().unique().tolist())
    except:
        pass

    print(f"   🔥 Total Arnaques trouvées : {len(bad_names)}")
    return list(bad_names)

# =========================================================
# 2. RÉCUPÉRATION DES FIABLES (DUMP CSV DOAJ)
# =========================================================
def get_legit_journals(target_count):
    print(f"\n2️⃣  Récupération des Revues Fiables (Cible : ~{target_count})...")
    print("   ⏳ Téléchargement du Dump DOAJ (20-30MB)...")
    
    try:
        # On télécharge le gros fichier d'un coup
        response = requests.get(DOAJ_CSV_URL, timeout=60)
        response.raise_for_status()
        
        # On le lit en mémoire
        csv_content = io.StringIO(response.content.decode('utf-8'))
        df_doaj = pd.read_csv(csv_content)
        
        # On extrait les titres
        if 'Journal title' in df_doaj.columns:
            all_titles = df_doaj['Journal title'].dropna().unique().tolist()
            print(f"   📚 Base DOAJ complète chargée : {len(all_titles)} revues.")
            
            # On en prend juste ce qu'il faut pour équilibrer (+ une petite marge)
            count_to_take = min(target_count, len(all_titles))
            selected = random.sample(all_titles, count_to_take)
            print(f"   ✨ Sélection aléatoire de {len(selected)} revues fiables.")
            return selected
        else:
            print("   ❌ Erreur structure CSV DOAJ.")
            return []
            
    except Exception as e:
        print(f"   ❌ Erreur téléchargement DOAJ : {e}")
        return []

# =========================================================
# 3. FUSION ET SAUVEGARDE
# =========================================================
def main():
    # A. Récupérer les méchants
    bad_list = get_predators()
    
    if not bad_list:
        print("❌ Arrêt critique : Impossible de trouver des prédateurs.")
        return

    # B. Récupérer les gentils (On vise le même nombre + 500 pour équilibrer)
    target_legit = len(bad_list) + 500
    good_list = get_legit_journals(target_legit)

    # C. Création du DataFrame
    print("\n3️⃣  Fusion et Nettoyage...")
    
    data = []
    for name in bad_list:
        data.append({'Titre': name, 'Est_Predateur': 1})
    
    for name in good_list:
        data.append({'Titre': name, 'Est_Predateur': 0})
        
    df = pd.DataFrame(data)
    
    # D. Priorité aux méchants en cas de doublon (Sécurité)
    # On trie par Est_Predateur décroissant (1 en premier), puis on supprime les doublons de Titre
    df = df.sort_values('Est_Predateur', ascending=False)
    df = df.drop_duplicates(subset=['Titre'], keep='first')
    
    # E. Mélange final
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # F. Sauvegarde
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ SUCCESS : Fichier généré '{OUTPUT_FILE}'")
    print("📊 Bilan du Dataset :")
    print(f"   🔴 Arnaques : {len(df[df['Est_Predateur']==1])}")
    print(f"   🟢 Fiables  : {len(df[df['Est_Predateur']==0])}")
    print(f"   ∑  TOTAL    : {len(df)}")

if __name__ == "__main__":
    main()