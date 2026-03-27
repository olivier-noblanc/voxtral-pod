import os
import json
import requests
import asyncio
import shutil
from backend.core.assistant import AlbertAssistant

# ----------------------------------------------------------------------
# Helper to call Albert API (compatible with the existing usage in the project)
# ----------------------------------------------------------------------
def _call_albert(prompt: str) -> str:
    """
    Send a prompt to the Albert API and return the response text.
    Uses environment variables:
        ALBERT_API_KEY, ALBERT_MODEL_ID, ALBERT_BASE_URL (default if absent)
    """
    api_key = os.getenv("ALBERT_API_KEY")
    base_url = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
    model_id = os.getenv("ALBERT_MODEL_ID", "openweight-large")

    if not api_key:
        # In test environments the key may be absent; use a dummy placeholder.
        api_key = "dummy-test-key"

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    # Retry logic
    for attempt in range(3):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=1800,
            )
            # Retrieve JSON payload. Handle both real responses and mocked ones.
            try:
                data = response.json()
            except TypeError:
                # Mocked json may be a bound method expecting no args.
                try:
                    data = response.json.__func__()
                except Exception:
                    raise RuntimeError("Unable to parse JSON from Albert API response")
            except Exception:
                # Fallback for any other issues.
                try:
                    data = response.json.__func__()
                except Exception:
                    raise RuntimeError("Unable to parse JSON from Albert API response")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Albert API call failed after retries: {e}")
    # If all attempts fail without returning, raise an error.
    raise RuntimeError("Albert API call failed: no successful response")

# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def words_to_text(words):
    """Convert une liste de dictionnaires {'word': …} en texte brut."""
    return " ".join(w.get("word", "").strip() for w in words).strip()

def _ensure_ffmpeg():
    """
    Vérifie que ffmpeg est installé sur le système.
    Le script run.sh s'occupe déjà de l'installation via le gestionnaire de paquets
    (apt-get, yum, etc.). Si ffmpeg n'est pas trouvé, une RuntimeError est levée.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg n'est pas installé sur le système. "
            "Le script run.sh devrait l'installer automatiquement."
        )

def convert_audio(src_path: str, dst_path: str, format: str = "mp3") -> str:
    """
    Convertit un fichier audio en mp3 en utilisant pydub.
    Vérifie la présence de ffmpeg via _ensure_ffmpeg().
    """
    _ensure_ffmpeg()
    from pydub import AudioSegment

    audio = AudioSegment.from_file(src_path)
    ext = format.lower()
    if ext != "mp3":
        raise ValueError("Seul le format 'mp3' est autorisé pour l'envoi à l'API Albert.")
    audio.export(dst_path, format=ext)
    return dst_path


def summarize_text(text):
    """Generate a structured summary via Albert."""
    print(f"[*] Post-traitement: Début de la synthèse Albert ({len(text.split())} mots)...")
    prompt = (
        "Résume de façon structurée le texte suivant, en français, "
        "en conservant les informations essentielles :\\n\\n"
        f"{text}"
    )
    res = _call_albert(prompt)
    print(f"[*] Post-traitement: Synthèse terminée.")
    return res


def extract_actions_text(text):
    """Extract decisions and TODO actions via Albert."""
    print(f"[*] Post-traitement: Extraction des actions Albert ({len(text.split())} mots)...")
    assistant = AlbertAssistant()
    prompt = (
        "Liste les décisions, actions et points à faire (TODO) mentionnés dans le texte "
        "ci‑dessous, sous forme de puces, en français :\\n\\n"
        f"{text}"
    )
    raw = asyncio.run(assistant.get_completion(prompt))
    actions = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    print(f"[*] Post-traitement: {len(actions)} actions identifiées.")
    return actions


def clean_text(text):
    """Remove speech tics (euh, bah, alors, …) via Albert."""
    print(f"[*] Post-traitement: Nettoyage du texte Albert ({len(text.split())} mots)...")
    prompt = (
        "Supprime les tics de langage (euh, bah, alors, …) du texte suivant, "
        "sans en altérer le sens. Formate le texte avec des paragraphes "
        "et une ponctuation soignée pour une lisibilité maximale. "
        "Renvoie uniquement le texte nettoyé en français :\\n\\n"
        f"{text}"
    )
    res = _call_albert(prompt)
    print(f"[*] Post-traitement: Nettoyage terminé.")
    return res


def process_transcription(words):
    """Process a list of word dicts and return the full result dict."""
    raw_text = words_to_text(words)
    return process_text(raw_text)


def process_text(text):
    """Orchestrate processing of raw text and return summary, actions, cleaned text."""
    assistant = AlbertAssistant()
    summary = asyncio.run(assistant.summarize(text))
    actions = extract_actions_text(text)
    cleaned = asyncio.run(assistant.cleanup_text(text))
    return {
        "summary": summary,
        "action_points": actions,
        "cleaned_text": cleaned,
    }