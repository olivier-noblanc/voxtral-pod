import os
import json
import requests
import asyncio
import shutil
import logging
from backend.core.assistant import AlbertAssistant

# Ensure NNPACK is disabled for postprocess operations
os.environ.setdefault("DISABLE_NNPACK", "1")

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper to call Albert API (compatible with the existing usage in the project)
# ----------------------------------------------------------------------
async def _call_albert(prompt: str) -> str:
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
        "max_tokens": 8192,  # Maximum standard output limit for most modern models
        "temperature": 0.0,
    }

    # Retry logic with exponential backoff
    for attempt in range(3):
        response = None
        if attempt > 0:
            await asyncio.to_thread(asyncio.sleep, 2**attempt)
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=1800,
            )
            
            # Check for HTTP errors before parsing
            response.raise_for_status()
            
            # Retrieve JSON payload simply
            try:
                data = response.json()
            except Exception as e:
                # If json() is an attribute (mocked incorrectly) instead of a method
                if not hasattr(response, "json") or not callable(response.json):
                    data = response.json if hasattr(response, "json") else {}
                else:
                    raise RuntimeError(f"Unable to parse JSON from Albert API response: {e}. Raw content: {response.text[:200]}")
            
            if not isinstance(data, dict):
                 raise RuntimeError(f"Albert API returned invalid format: expected dict, got {type(data)}")

            choices = data.get("choices")
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                 logger.error(f"Albert API response missing 'choices': {data}")
                 raise RuntimeError(f"Albert API response missing 'choices'")

            message = choices[0].get("message", {})
            content = message.get("content")
            
            if content is None:
                refusal = message.get("refusal")
                if refusal:
                    logger.error(f"Albert API refused to answer: {refusal}")
                    raise RuntimeError(f"Albert API refusal: {refusal}")
                logger.error(f"Albert API message exists but has no content. Full data: {data}")
                return ""
            
            return content
        except requests.exceptions.HTTPError:
            if attempt == 2:
                status_code = response.status_code if response is not None else "N/A"
                text = response.text[:200] if response is not None else "N/A"
                raise RuntimeError(f"Albert API HTTP error {status_code}: {text}")
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Albert API call failed after retries: {e}")
    
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


async def summarize_text(text):
    if not text.strip():
        return ""
    print(f"[*] Post-traitement: Début de la synthèse Albert ({len(text.split())} mots)...")
    prompt = (
        "Résume de façon structurée le texte suivant, en français, "
        "en conservant les informations essentielles :\\n\\n"
        f"{text}"
    )
    res = await _call_albert(prompt)
    print(f"[*] Post-traitement: Synthèse terminée.")
    return res


async def extract_actions_text(text):
    """Extract decisions and TODO actions via Albert."""
    print(f"[*] Post-traitement: Extraction des actions Albert ({len(text.split())} mots)...")
    assistant = AlbertAssistant()
    prompt = (
        "Liste les décisions, actions et points à faire (TODO) mentionnés dans le texte "
        "ci‑dessous, sous forme de puces, en français :\\n\\n"
        f"{text}"
    )
    raw = await assistant.get_completion(prompt)
    actions = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    print(f"[*] Post-traitement: {len(actions)} actions identifiées.")
    return actions


async def clean_text(text):
    if not text.strip():
        return ""
    print(f"[*] Post-traitement: Nettoyage du texte Albert ({len(text.split())} mots)...")
    prompt = (
        "Supprime les tics de langage (euh, bah, alors, ...) du texte suivant, "
        "sans en altérer le sens. Formate le texte avec des paragraphes "
        "et une ponctuation soignée pour une lisibilité maximale. "
        "Renvoie uniquement le texte nettoyé en français :\n\n"
        f"{text}"
    )
    logger.debug(f"Prompt Albert (Clean) - Longueur: {len(prompt)}, Début: {prompt[:100]}...")
    res = await _call_albert(prompt)
    if not res:
        logger.warning("Albert API (Clean) a renvoyé un résultat vide.")
    else:
        logger.info(f"Albert API (Clean) a renvoyé {len(res)} caractères.")
    
    print(f"[*] Post-traitement: Nettoyage terminé.")
    return res


async def process_transcription(words):
    """Process a list of word dicts and return the full result dict."""
    raw_text = words_to_text(words)
    return await process_text(raw_text)


async def process_text(text):
    """Orchestrate processing of raw text and return summary, actions, cleaned text."""
    assistant = AlbertAssistant()
    summary = await assistant.summarize(text)
    actions = await extract_actions_text(text)
    cleaned = await assistant.cleanup_text(text)
    return {
        "summary": summary,
        "action_points": actions,
        "cleaned_text": cleaned,
    }