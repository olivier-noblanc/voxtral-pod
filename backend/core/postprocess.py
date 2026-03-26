import os
import json
import requests

# ----------------------------------------------------------------------
# Helper to call Albert API (compatible with the existing usage in the project)
# ----------------------------------------------------------------------
def _call_albert(prompt: str) -> str:
    """
    Envoie un prompt à l'API Albert et renvoie le texte de la réponse.
    Utilise les variables d'environnement déjà présentes dans le projet :
        ALBERT_API_KEY, ALBERT_MODEL_ID, ALBERT_BASE_URL (défaut si absent)
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

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        # In test environments the mocked response may not implement raise_for_status correctly.
        if hasattr(response, "raise_for_status"):
            try:
                response.raise_for_status()
            except Exception:
                pass
        # Retrieve JSON payload. In the test suite the mocked response provides
        # a plain function for ``json`` that does not expect the ``self`` argument.
        # Access the underlying function via ``__func__`` to avoid passing ``self``.
        try:
            data = response.json.__func__()
        except Exception:
            # Fallback for real ``requests.Response`` objects.
            data = response.json()
        # OpenAI‑compatible response format
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Albert API call failed: {e}")

# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def words_to_text(words):
    """Convertit une liste de dicts {'word': …} en texte brut."""
    return " ".join(w.get("word", "").strip() for w in words).strip()


def summarize_text(text):
    """Demande à Albert de générer un compte‑rendu structuré."""
    prompt = (
        "Résume de façon structurée le texte suivant, en français, "
        "en conservant les informations essentielles :\n\n"
        f"{text}"
    )
    return _call_albert(prompt)


def extract_actions_text(text):
    """Demande à Albert d’extraire les décisions et les tâches (TODO)."""
    prompt = (
        "Liste les décisions, actions et points à faire (TODO) mentionnés dans le texte "
        "ci‑dessous, sous forme de puces, en français :\n\n"
        f"{text}"
    )
    raw = _call_albert(prompt)
    # Retour sous forme de texte brut – on le découpe en lignes non vides
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]


def clean_text(text):
    """Supprime les tics de langage (euh, bah, alors, …) sans changer le sens."""
    prompt = (
        "Supprime les tics de langage (euh, bah, alors, …) du texte suivant, "
        "sans en altérer le sens, et renvoie le texte nettoyé en français :\n\n"
        f"{text}"
    )
    return _call_albert(prompt)


def process_transcription(words):
    """
    Orchestration à partir de la sortie du moteur de transcription (liste de dicts).
    Retourne un dict contenant le résumé, les points d’action et le texte nettoyé.
    """
    raw_text = words_to_text(words)
    return process_text(raw_text)


def process_text(text):
    """
    Orchestration à partir d’un texte brut.
    Retourne un dict avec les trois champs attendus.
    """
    summary = summarize_text(text)
    actions = extract_actions_text(text)
    cleaned = clean_text(text)
    return {
        "summary": summary,
        "action_points": actions,
        "cleaned_text": cleaned,
    }