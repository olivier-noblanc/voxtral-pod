import keyring
import requests

from backend.core.albert_rate_limiter import albert_rate_limiter


def test_real_albert_quota():
    """
    Test d'intégration REEL. 
    Interroge l'API Albert avec la clé du keyring.
    """
    print("\n--- TEST D'INTEGRATION ALBERT QUOTA ---")
    
    # 1. Récupération de la clé
    key = keyring.get_password("albert_api", "default")
    if not key:
        print("❌ Erreur : Aucune clé trouvée dans le keyring (albert_api, default).")
        return
    
    print(f"✅ Clé trouvée (longueur: {len(key)})")
    
    # 2. Configuration du rate limiter
    albert_rate_limiter.albert_api_key = key
    albert_rate_limiter._has_valid_api_key = True
    
    # 3. Appel de l'API
    print("⏳ Interpellation de l'API Albert (me/usage & me/info)...")
    headers = {"Authorization": f"Bearer {key}"}
    
    # Debug: voir la structure exacte
    r_debug = requests.get(f"{albert_rate_limiter.albert_base_url}/me/usage", headers=headers, timeout=5)
    if r_debug.status_code == 200:
        d = r_debug.json()
        print("🔍 Structure des enregistrements d'usage :")
        if "data" in d and len(d["data"]) > 0:
            for i, record in enumerate(d["data"][:10]):
                print(f"--- Record {i+1} ---")
                print(f"Model: {record.get('model')}, Endpoint: {record.get('endpoint')}")
                # print(json.dumps(record, indent=2))
        else:
            print("   (Aucune donnée d'usage trouvée)")
    
    albert_rate_limiter._last_quota_update = 0 # Forcer l'update
    albert_rate_limiter.update_quota_info()
    
    # 4. Affichage des résultats
    info = albert_rate_limiter.get_status_info()
    print("📊 Résultat détaillé du Quota :")
    print(f"   ASR (Whisper) : {info['quota_asr_usage']} / {info['quota_limit']}")
    print(f"   LLM (Chat)     : {info['quota_llm_usage']}")
    print(f"   Usage Global   : {info['quota_usage']}")
    
    if info['quota_limit'] > 0:
        print("✅ Succès : L'API a répondu correctement.")
    else:
        print("⚠️ Attention : L'API a répondu mais la limite est à 0 (vérifier le format de réponse).")

if __name__ == "__main__":
    test_real_albert_quota()
