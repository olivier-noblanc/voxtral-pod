import asyncio
from unittest.mock import AsyncMock, patch

from backend.core.assistant import AlbertAssistant
from backend.core.postprocess import process_text, process_transcription


def test_process_text_uses_albertassistant_methods() -> None:
    async def run():
        with patch.object(AlbertAssistant, "summarize", new_callable=AsyncMock, return_value="Résumé mocké") as mock_summarize, \
             patch.object(AlbertAssistant, "cleanup_text", new_callable=AsyncMock, return_value="Texte nettoyé mocké") as mock_cleanup, \
             patch("backend.core.postprocess.extract_actions_text", new_callable=AsyncMock, return_value=["Action 1", "Action 2"]):
            
            result = await process_text("texte d'exemple")
            
            assert result["summary"] == "Résumé mocké"
            assert result["cleaned_text"] == "Texte nettoyé mocké"
            mock_summarize.assert_called_once()
            mock_cleanup.assert_called_once()

    asyncio.run(run())

def test_process_transcription_uses_albertassistant_methods() -> None:
    async def run():
        with patch.object(AlbertAssistant, "summarize", new_callable=AsyncMock, return_value="Résumé mocké") as mock_summarize, \
             patch.object(AlbertAssistant, "cleanup_text", new_callable=AsyncMock, return_value="Texte nettoyé mocké") as mock_cleanup, \
             patch("backend.core.postprocess.extract_actions_text", new_callable=AsyncMock, return_value=["Action 1", "Action 2"]):
            
            words = [{"word": "Bonjour"}, {"word": "le"}, {"word": "monde"}]
            result = await process_transcription(words)
            
            assert result["summary"] == "Résumé mocké"
            assert result["cleaned_text"] == "Texte nettoyé mocké"
            mock_summarize.assert_called_once()
            mock_cleanup.assert_called_once()

    asyncio.run(run())
