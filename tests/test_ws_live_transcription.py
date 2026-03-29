"""
Tests du pipeline WebSocket live complet.
Vérifie que process_audio_queue() envoie les bons messages JSON au WebSocket.

Convention live.py :
  - is_speech()          → True  = parole active (activation)
  - check_deactivation() → True  = encore de la parole (pas de fin)
                            False = silence → silence_chunks_count ++
"""
import asyncio
import numpy as np

from backend.core.live import LiveSession
from typing import cast, Any
from fastapi import WebSocket
from backend.core.vad import VADManager

# pylint: disable=protected-access

# ---- Dummies ----

class DummyWebSocket:
    """Mock pour WebSocket FastAPI capturant les messages envoyés."""
    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        """Enregistre le JSON envoyé."""
        self.messages.append(data)


class DummyTranscriptionEngine:
    """Mock pour le moteur de transcription Whisper/Albert."""
    def transcribe(self, _audio_np):
        """Renvoie toujours 'bonjour monde'."""
        return ([{"word": "bonjour"}, {"word": "monde"}], None)


class DummyEngine:
    """Mock pour le wrapper de modèle Engine."""
    def __init__(self, model_id="test"):
        self.model_id = model_id
        self.transcription_engine = DummyTranscriptionEngine()


class DummyEmptyEngine:
    """Mock pour le wrapper de modèle Engine renvoyant du vide."""
    def __init__(self, model_id="empty"):
        self.model_id = model_id
        self.transcription_engine = type('Dummy', (), {'transcribe': lambda _self, _x: ([], None)})()


# ---- VADs de test ----

class SpeechThenSilenceVAD:
    """
    Retourne True (parole) pour les N premiers is_speech(),
    puis False pour tous les check_deactivation() suivants (= silence).

    Le cycle est :
      1. live.py appelle is_speech() → True (activation de la parole)
      2. live.py appelle check_deactivation() sur les chunks suivants
         → dès que speech_chunks épuisés, retourne False = silence
    """
    def __init__(self, speech_chunks: int = 3):
        self._n = speech_chunks          # nb de chunks de « parole »
        self._speech_calls = 0           # compteur is_speech()

    def reset_states(self):
        """Réinitialise les compteurs."""
        self._speech_calls = 0

    async def is_speech(self, _audio_bytes) -> bool:
        """Simule le début de parole."""
        self._speech_calls += 1
        return self._speech_calls <= self._n

    async def check_deactivation(self, _audio_bytes) -> bool:
        """Simule la fin de parole immédiate une fois le quota atteint."""
        # Retourne False immédiatement = silence dès que le quota est dépassé
        return False


class AlwaysSpeechVAD:
    """Jamais de silence — pour déclencher des partials."""
    def reset_states(self):
        """Dummy reset."""

    async def is_speech(self, _b) -> bool:
        """Toujours parole."""
        return True

    async def check_deactivation(self, _b) -> bool:
        """Jamais silence."""
        return True


# ---- Helper PCM ----

def make_pcm_chunk(duration_ms: int = 100, amplitude: float = 0.5) -> bytes:
    """Génère un chunk audio PCM16 de test."""
    samples = int(16000 * duration_ms / 1000)
    arr = (np.ones(samples, dtype=np.float32) * amplitude * 32767).astype(np.int16)
    return arr.tobytes()


# ---- Tests ----

async def _test_ws_receives_final_sentence():
    """
    3 chunks où is_speech=True (activation),
    puis 6 chunks où check_deactivation=False (silence).
    silence_chunks_threshold = 5 → 6 chunks déclenchent VAD END.
    Attendu : au moins 1 message final=True sur le WebSocket.
    """
    ws = DummyWebSocket()
    session = LiveSession(engine=DummyEngine(), websocket=cast(WebSocket, ws), client_id="user_t01")
    session.vad = cast(VADManager, SpeechThenSilenceVAD(speech_chunks=3))

    proc = asyncio.create_task(session.process_audio_queue())

    speech = make_pcm_chunk(100, amplitude=0.5)
    silence = make_pcm_chunk(100, amplitude=0.0)

    # 3 chunks → is_speech=True → is_speaking=True
    for _ in range(3):
        await session.audio_queue.put(speech)
    # 6 chunks → check_deactivation=False → silence_chunks_count monte jusqu'au seuil
    for _ in range(6):
        await session.audio_queue.put(silence)

    await session.audio_queue.put(None)
    await proc

    assert ws.messages, (
        f"Aucun message WS reçu. sentences={session._sentences}, "
        f"audio_chunks={len(session.full_session_audio)}"
    )
    final_msgs = [m for m in ws.messages if m.get("final") is True]
    assert final_msgs, f"Aucun message final. messages={ws.messages}"

    msg = final_msgs[0]
    assert msg["type"] == "sentence"
    assert "bonjour monde" in msg["text"], f"Texte inattendu: {msg['text']}"
    assert msg["speaker"] == "Speaker"
    assert msg["sentence_index"] >= 0
    assert "total_bytes_received" in msg


async def _test_ws_receives_partial_sentence():
    """
    5 chunks de 500ms = 80 000 bytes > seuil partial 64 000.
    VAD toujours actif → pas de fin → seuls des partials.
    Attendu : au moins 1 message final=False.
    """
    ws = DummyWebSocket()
    session = LiveSession(engine=DummyEngine(), websocket=cast(WebSocket, ws), client_id="user_t02")
    session.vad = cast(VADManager, AlwaysSpeechVAD())

    proc = asyncio.create_task(session.process_audio_queue())

    chunk = make_pcm_chunk(500, amplitude=0.5)
    for _ in range(5):   # 5 × 16 000 = 80 000 bytes > 64 000 (partial_interval)
        await session.audio_queue.put(chunk)

    await session.audio_queue.put(None)
    await proc

    partials = [m for m in ws.messages if m.get("final") is False]
    assert partials, f"Aucun partial reçu. messages={ws.messages}"
    assert "bonjour monde" in partials[0]["text"]
    assert partials[0]["type"] == "sentence"


def test_ws_receives_final_sentence():
    """Le WebSocket doit recevoir au moins une phrase finale (final=True)."""
    asyncio.run(_test_ws_receives_final_sentence())


def test_ws_receives_partial_sentence():
    """Le WebSocket doit recevoir au moins un partial (final=False)."""
    asyncio.run(_test_ws_receives_partial_sentence())


async def _test_ws_receives_empty_final_sentence():
    """
    Vérifie qu'un segment qui ne contient QUE du silence (ou que l'IA ignore) 
    envoie quand même un message final=True pour clore la ligne côté frontend.
    """
    ws = DummyWebSocket()
    session = LiveSession(engine=DummyEmptyEngine(), websocket=cast(WebSocket, ws), client_id="user_empty")
    session.vad = cast(VADManager, SpeechThenSilenceVAD(speech_chunks=0)) # Directement silence après audit
    session.is_speaking = True # Simulation : on croyait parler
    
    proc = asyncio.create_task(session.process_audio_queue())
    silence = make_pcm_chunk(100, amplitude=0.0)
    for _ in range(6):
        await session.audio_queue.put(silence)
    await session.audio_queue.put(None)
    await proc

    final_msgs = [m for m in ws.messages if m.get("final") is True]
    assert final_msgs, "Un message final avec texte vide DOIT être envoyé pour clore le segment"
    assert final_msgs[0]["text"] == ""


def test_ws_receives_empty_final_sentence():
    """Pytest entry for empty final sentence test."""
    asyncio.run(_test_ws_receives_empty_final_sentence())
