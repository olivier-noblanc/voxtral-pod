"""
Tests pour vérifier l'état et la télémétrie d'une session Live.
"""
import asyncio
from backend.core.live import LiveSession

# Dummy VAD manager that always reports speech and never deactivates
class DummyVADManager:
    """Mock pour VADManager évitant l'instanciation de modèles réels."""
    def __init__(self, *args, **kwargs):
        pass

    def reset_states(self):
        """Réinitialisation simulée."""

    async def is_speech(self, _audio_bytes):
        """Simule la détection de parole Toujours active."""
        return True

    async def check_deactivation(self, _audio_bytes):
        """Simule la non-désactivation de la parole."""
        return False

# Patch the VADManager import in LiveSession to use the dummy one
# pylint: disable=protected-access
LiveSession.__init__.__globals__['VADManager'] = DummyVADManager

# Dummy WebSocket that records sent JSON messages
class DummyWebSocket:
    """Mock pour WebSocket FastAPI capturant les messages envoyés."""
    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        """Enregistre le JSON envoyé."""
        self.messages.append(data)

# Dummy transcription engine that returns a fixed word list (synchronous)
class DummyTranscriptionEngine:
    """Mock pour le moteur de transcription Whisper/Albert."""
    def transcribe(self, _audio_np):
        """Renvoie toujours 'hello'."""
        # Return a single word "hello" for any input
        return ([{"word": "hello"}], None)

# Dummy engine wrapper expected by LiveSession
class DummyEngine:
    """Mock pour le wrapper de modèle Engine."""
    def __init__(self):
        self.model_id = "test"
        self.transcription_engine = DummyTranscriptionEngine()

# Async helper that contains the original test logic
async def _run_live_session_telemetry():
    """
    Logique de test asynchrone pour vérifier les compteurs d'octets
    et la structure des messages WebSocket.
    """
    # Arrange
    dummy_ws = DummyWebSocket()
    dummy_engine = DummyEngine()
    session = LiveSession(engine=dummy_engine, websocket=dummy_ws, client_id="test_client")

    # Simulate receiving two audio chunks
    chunk1 = b"audio_chunk_1"
    chunk2 = b"audio_chunk_2"
    session._total_bytes_received = 0  # start from zero

    # Manually feed chunks as process_audio_queue would do
    session.full_session_audio.append(chunk1)
    session._total_bytes_received += len(chunk1)

    session.full_session_audio.append(chunk2)
    session._total_bytes_received += len(chunk2)

    # Simulate end of a sentence (final=True) by directly calling the private method
    await session._transcribe_segment(b"".join([chunk1, chunk2]), final=True)

    # Assert that total bytes received matches the sum of chunk sizes
    assert session._total_bytes_received == len(chunk1) + len(chunk2)

    # Assert that a sentence was stored and the index was incremented
    assert session._sentences == ["hello"]
    assert session._sentence_index == 1

    # Verify that the websocket message contains the telemetry metadata
    assert len(dummy_ws.messages) == 1
    msg = dummy_ws.messages[0]
    assert msg["type"] == "sentence"
    assert msg["text"] == "hello"
    assert msg["final"] is True
    assert msg["sentence_index"] == 0
    assert msg["total_bytes_received"] == session._total_bytes_received

# Synchronous test entry point – runs the async helper via asyncio.run
def test_live_session_telemetry():
    """Point d'entrée pytest pour le test de télémétrie live."""
    asyncio.run(_run_live_session_telemetry())