import os

# Ensure NNPACK is disabled for assistant operations
os.environ.setdefault("DISABLE_NNPACK", "1")

class AlbertAssistant:
    def __init__(self) -> None:
        self.api_key = os.getenv("ALBERT_API_KEY")
        self.base_url = "https://albert.api.etalab.gouv.fr/v1"
        self.model_id = "openweight-large" # Specified by user

    async def get_completion(
        self,
        prompt: str,
        system_message: str = "Tu es un assistant expert en analyse de transcriptions de réunions."
    ) -> str:
        """Call Albert LLM for completion."""
        # Utilise la fonction de helper déjà mockée dans le module postprocess.
        # Le mock dans les tests cible ``backend.core.postprocess.requests.post``,
        # donc appeler ``_call_albert`` garantit que le mock est appliqué.
        # Import locally to avoid circular import with postprocess
        from backend.core import postprocess
        return await postprocess._call_albert(prompt)

    async def summarize(self, text: str) -> str:
        from backend.core import postprocess
        return await postprocess.summarize_text(text)

    async def cleanup_text(self, text: str) -> str:
        from backend.core import postprocess
        return await postprocess.clean_text(text)
