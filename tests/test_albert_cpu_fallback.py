import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.core.albert_rate_limiter import albert_rate_limiter
from backend.core.engine import SotaASR


@pytest.mark.asyncio
async def test_albert_fallback_does_not_crash_with_handle_error() -> None:
    """
    Test de non-régression pour éviter le crash (NoneType has no attribute '_handle')
    lors du fallback. Vérifie que TranscriptionEngine.load() initialise 
    correctement le modèle de repli local au lieu de Vosk (ou de crasher).
    """

    # Force state and bypass Auto-Mock by setting TESTING="0" temporarily
    with patch.dict("os.environ", {"TESTING": "0", "ALBERT_API_KEY": "fake_key_123"}):
        engine = SotaASR(model_id="albert")
        engine.transcription_engine.albert_api_key = "fake_key_123"
        albert_rate_limiter._has_valid_api_key = True
        albert_rate_limiter.albert_api_key = "fake_key_123"

    engine.load()

    # Déclencher le rate limit artificiellement
    albert_rate_limiter._consecutive_429 = 5
    assert albert_rate_limiter.should_use_cpu_fallback_mode() is True

    # Préparer un audio de silence
    audio_np = np.zeros(16000, dtype=np.float32)

    # Patch l'instantiation de Whisper pour éviter le téléchargement de 3 Go dans les runners CI
    with patch("faster_whisper.WhisperModel") as mock_whisper:
        # Mock the transcription result to avoid errors in .transcribe()
        mock_model_instance = MagicMock()
        
        # Le retour attendu d'un appel à Whisper: segments et info
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.words = []
        mock_segment.text = "test fallback ok"
        
        mock_info = MagicMock()
        mock_info.duration = 1.0
        
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper.return_value = mock_model_instance

        # Cet appel crashait auparavant avec :
        # AttributeError: 'NoneType' object has no attribute '_handle'
        # dû au chargement par défaut (et aveugle) de Vosk.
        words, duration, used_fallback, raw_data = await asyncio.to_thread(
            engine.transcription_engine.transcribe, audio_np
        )

        assert used_fallback is True
        assert engine.transcription_engine.model_id == "cpu"
        
        # Whisper a dû être appelé (via le mock)
        mock_whisper.assert_called_once()
        mock_model_instance.transcribe.assert_called_once()
