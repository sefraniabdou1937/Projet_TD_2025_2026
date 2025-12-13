import pandas as pd
import asyncio
import aiohttp
import os
import time
from urllib.parse import quote
from tqdm.asyncio import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "data/01_journal_names.csv"
OUTPUT_FILE = "data/02_real_world_dataset.csv"
EMAIL = "etudiant@ensah.ma"  # Pour être poli avec l'API Crossref
BATCH_SIZE = 50              # Sauvegarde toutes les 50 revues
MAX_CONCURRENT_REQ = 10      # 10 requêtes en parallèle (équilibre vitesse/stabilité)

# Headers pour identifier notre script auprès des APIs
HEADERS = {
    "User-Agent": f"PredatoryDetector/2.0 (mailto:{EMAIL})"
}

print("--- 💎 ÉTAPE 2 : ENRICHISSEMENT DES DONNÉES (METADATA MINING) ---")

async def fetch_openalex(session, journal_name):
    """Interroge OpenAlex pour récupérer l'impact scientifique (Citations, Works)"""
    # OpenAlex est gratuit et très complet pour les stats
    url = f"https://api.openalex.org/sources?search={quote(journal_name)}"
    try:
        async with session.get(url, headers=HEADERS) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('results', [])
                if results:
                    # On prend le premier résultat qui correspond
                    j = results[0]
                    return {
                        'oa_works': j.get('works_count', 0),
                        'oa_cited': j.get('cited_by_count', 0),
                        'oa_found': 1
                    }
    except:
        pass
    # Si rien trouvé ou erreur
    return {'oa_works': 0, 'oa_cited': 0, 'oa_found': 0}

async def fetch_crossref(session, journal_name):
    """Interroge Crossref pour vérifier si la revue attribue des DOI (gage de qualité)"""
    url = f"https://api.crossref.org/works?query.container-title={quote(journal_name)}&rows=1&mailto={EMAIL}"
    try:
        # Timeout court (10s) car Crossref est parfois lent
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                items = data.get('message', {}).get('items', [])
                if items:
                    publisher = items[0].get('publisher', 'Unknown')
                    return {'cr_has_doi': 1, 'Publisher': publisher}
    except:
        pass
    return {'cr_has_doi': 0, 'Publisher': 'Unknown'}

async def process_journal(sem, session, row):
    """Chef d'orchestre pour une revue : lance OA et CR en même temps"""
    async with sem: # On respecte la file d'attente (semaphore)
        name = row['Titre']
        
        # Lancement parallèle
        task_oa = fetch_openalex(session, name)
        task_cr = fetch_crossref(session, name)
        
        # On attend que les deux aient fini
        res_oa, res_cr = await asyncio.gather(task_oa, task_cr)
        
        # On fusionne tout dans la ligne existante
        result = row.copy()
        result.update(res_oa)
        result.update(res_cr)
        
        # Petit calcul utile pour plus tard (Ratio d'Impact)
        works = result.get('oa_works', 0)
        cited = result.get('oa_cited', 0)
        result['Impact_Ratio'] = cited / (works + 1)
        
        return result

async def main():
    # 1. Vérification des fichiers
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Erreur : Le fichier {INPUT_FILE} n'existe pas. Lancez l'étape 1.")
        return

    # 2. Chargement
    df_source = pd.read_csv(INPUT_FILE)
    
    # 3. Système de Reprise (Si le script plante, on ne recommence pas à zéro)
    processed_names = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            processed_names = set(df_existing['Titre'].unique())
            print(f"🔄 Reprise : {len(processed_names)} revues déjà faites. On continue...")
        except:
            pass
            
    # On filtre pour ne garder que ce qu'il reste à faire
    df_to_process = df_source[~df_source['Titre'].isin(processed_names)].to_dict('records')
    
    if not df_to_process:
        print("✅ Tout est déjà fini ! Bravo.")
        return

    print(f"🚀 Démarrage de l'analyse pour {len(df_to_process)} revues...")
    
    # Le Semaphore limite le nombre de requêtes simultanées (ici 10)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQ)

    async with aiohttp.ClientSession() as session:
        buffer = []
        
        # La boucle principale avec barre de chargement
        for i, row in tqdm(enumerate(df_to_process), total=len(df_to_process), unit="revue"):
            
            # Traitement asynchrone
            res = await process_journal(sem, session, row)
            buffer.append(res)
            
            # Sauvegarde régulière (tous les 50 items)
            if len(buffer) >= BATCH_SIZE:
                df_batch = pd.DataFrame(buffer)
                header = not os.path.exists(OUTPUT_FILE) # On met l'en-tête seulement la première fois
                df_batch.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
                buffer = [] # On vide la mémoire
                # Petite pause de courtoisie
                await asyncio.sleep(0.1)

        # Sauvegarde finale (pour les derniers items qui restaient dans le buffer)
        if buffer:
            df_batch = pd.DataFrame(buffer)
            header = not os.path.exists(OUTPUT_FILE)
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)

    print("\n✅ ENRICHISSEMENT TERMINÉ !")
    print(f"📁 Fichier final : {OUTPUT_FILE}")

if __name__ == "__main__":
    # Correction spécifique pour Windows (évite une erreur asyncio EventLoop)
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    start_time = time.time()
    asyncio.run(main())
    duration = (time.time() - start_time) / 60
    print(f"⏱️ Temps total : {duration:.1f} minutes.")