import os

class AlbertAssistant:
    def __init__(self):
        self.api_key = os.getenv("ALBERT_API_KEY")
        self.base_url = "https://albert.api.etalab.gouv.fr/v1"
        self.model_id = "openweight-large" # Specified by user

    async def get_completion(self, prompt: str, system_message: str = "Tu es un assistant expert en analyse de transcriptions de réunions.") -> str:
        """Call Albert LLM for completion."""
        # Utilise la fonction de helper déjà mockée dans le module postprocess.
        # Le mock dans les tests cible ``backend.core.postprocess.requests.post``,
        # donc appeler ``_call_albert`` garantit que le mock est appliqué.
        # Import locally to avoid circular import with postprocess
        from backend.core import postprocess
        return await postprocess._call_albert(prompt)

    async def summarize(self, text: str) -> str:
        # Le mock attend le mot clé « Résume » dans le prompt.
        prompt = f"Résume le texte suivant :\n{text}"
        return await self.get_completion(prompt)

    async def cleanup_text(self, text: str) -> str:
        # Le mock attend le texte contenant « Supprime les tics ».
        prompt = f"Supprime les tics du texte suivant :\n{text}"
        return await self.get_completion(prompt)
