import argparse
import keyring

def main():
    parser = argparse.ArgumentParser(description="Voxtral CLI Tools")
    parser.add_argument("--set-key", help="Enregistrer une clé API dans le trousseau", choices=["albert_api_key"])
    parser.add_argument("--value", help="Valeur de la clé")

    args = parser.parse_args()

    if args.set_key and args.value:
        keyring.set_password("voxtral", args.set_key, args.value)
        print(f"[*] Clé {args.set_key} enregistrée avec succès dans le trousseau système.")
    else:
        print("[!] Usage: python backend/cli_tools.py --set-key albert_api_key --value YOUR_KEY")

if __name__ == "__main__":
    main()
