import os
import requests
import json

class AlbertAssistant:
    def __init__(self):
        self.api_key = os.getenv("ALBERT_API_KEY")
        self.base_url = "https://albert.api.etalab.gouv.fr/v1"
        self.model_id = "openweight-large" # Specified by user

    async def get_completion(self, prompt: str, system_message: str = "Tu es un assistant expert en analyse de transcriptions de réunions.") -> str:
        """Call Albert LLM for completion."""
        if not self.api_key:
            return "Erreur : Clé API Albert non configurée (ALBERT_API_KEY)."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        try:
            # Using requests in a thread since it's sync, but we are in an async method
            import asyncio
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )

            if response.status_code != 200:
                return f"Erreur API Albert ({response.status_code}) : {response.text}"

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Exception lors de l'appel LLM : {str(e)}"

    async def summarize(self, text: str) -> str:
        prompt = f"Voici la transcription d'une réunion. Merci d'en faire un résumé structuré, d'en extraire les points clés et les éventuelles actions à entreprendre (TODO list).\n\nTRANSCRIPTION :\n{text}"
        return await self.get_completion(prompt)

    async def cleanup_text(self, text: str) -> str:
        prompt = f"Merci de nettoyer cette transcription en supprimant les tics de langage (euh, bah, alors, etc.) et en corrigeant les fautes de frappe évidentes, tout en restant fidèle au sens original.\n\nTEXTE :\n{text}"
        return await self.get_completion(prompt)
