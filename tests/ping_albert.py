import os

import requests


def ping_albert() -> None:
    api_key = os.getenv("ALBERT_API_KEY")
    base_url = "https://albert.api.etalab.gouv.fr/v1"
    
    if not api_key:
        print("[!] Erreur: La variable d'environnement ALBERT_API_KEY n'est pas définie.")
        return

    print(f"[*] Test de connexion vers {base_url}...")
    try:
        # On tente de lister les modèles ou juste un GET sur l'init
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("[+] Succès ! L'API Albert répond correctement.")
            models = response.json()
            print(f"[ ] Modèles disponibles : {len(models.get('data', []))}")
        else:
            print(f"[!] Échec: Le serveur a répondu avec le statut {response.status_code}")
            print(f"[ ] Détails: {response.text}")
            
    except Exception as e:
        print("[!] Erreur critique lors de la connexion :")
        print(f"    Type: {type(e).__name__}")
        print(f"    Message: {e}")

if __name__ == "__main__":
    ping_albert()
