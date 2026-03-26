import pytest
from unittest.mock import patch
from backend.core.postprocess import process_text, process_transcription
from backend.core.assistant import AlbertAssistant

def test_process_text_uses_albertassistant_methods():
    # Patch AlbertAssistant methods to ensure they are called
    with patch.object(AlbertAssistant, "summarize", return_value="Résumé mocké") as mock_summarize, \
         patch.object(AlbertAssistant, "cleanup_text", return_value="Texte nettoyé mocké") as mock_cleanup, \
         patch("backend.core.postprocess.extract_actions_text", return_value=["Action 1", "Action 2"]):
        result = process_text("texte d'exemple")
        assert result["summary"] == "Résumé mocké"
        assert result["cleaned_text"] == "Texte nettoyé mocké"
        # Verify that the AlbertAssistant methods were invoked exactly once
        mock_summarize.assert_called_once()
        mock_cleanup.assert_called_once()

def test_process_transcription_uses_albertassistant_methods():
    # Patch AlbertAssistant methods to ensure they are called during transcription processing
    with patch.object(AlbertAssistant, "summarize", return_value="Résumé mocké") as mock_summarize, \
         patch.object(AlbertAssistant, "cleanup_text", return_value="Texte nettoyé mocké") as mock_cleanup, \
         patch("backend.core.postprocess.extract_actions_text", return_value=["Action 1", "Action 2"]):
        words = [{"word": "Bonjour"}, {"word": "le"}, {"word": "monde"}]
        result = process_transcription(words)
        assert result["summary"] == "Résumé mocké"
        assert result["cleaned_text"] == "Texte nettoyé mocké"
        mock_summarize.assert_called_once()
        mock_cleanup.assert_called_once()