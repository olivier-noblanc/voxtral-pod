"""
Tests de non-régression pour la robustesse de save_audio_file().
Vérifie que la méthode ne crashe pas quelque soit le type de retour
de transcription_engine.transcribe() : list[dict], tuple(str, _), str directe.
"""
import asyncio
import os
import tempfile
import numpy as np
import pytest

from backend.core.live import LiveSession


class _WebSocketDummy:
    async def send_json(self, data): pass

def _make_audio_data():
    samples = np.zeros(1600, dtype=np.int16)
    return samples.tobytes()

class _EngineWithStringTranscribe:
    model_id = "mock"
    class _Trans:
        def transcribe(self, audio_np):
            return "texte complet de test"
    transcription_engine = _Trans()

class _EngineWithTupleStringTranscribe:
    model_id = "albert"
    class _Trans:
        def transcribe(self, audio_np):
            return ("texte albert complet", None)
    transcription_engine = _Trans()

class _EngineWithWordListTranscribe:
    model_id = "whisper"
    class _Trans:
        def transcribe(self, audio_np):
            return ([{"word": "bonjour"}, {"word": "whisper"}], None)
    transcription_engine = _Trans()


def _build_session(engine, tmp_path, client_id="user_satest") -> LiveSession:
    ws = _WebSocketDummy()
    session = LiveSession(engine=engine, websocket=ws, client_id=client_id)
    session.full_session_audio = [_make_audio_data() for _ in range(5)]
    return session


def test_save_audio_file_string_return(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineWithStringTranscribe(), tmp_path)
    async def run():
        return await session.save_audio_file()
    result = asyncio.run(run())
    assert result is not None
    txt_name = result.replace(".wav", ".txt")
    txt_path = tmp_path / "live_audio" / txt_name
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == "texte complet de test"


def test_save_audio_file_tuple_string_return(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineWithTupleStringTranscribe(), tmp_path, "user_satest2")
    async def run():
        return await session.save_audio_file()
    result = asyncio.run(run())
    assert result is not None
    txt_name = result.replace(".wav", ".txt")
    txt_path = tmp_path / "live_audio" / txt_name
    assert "albert" in txt_path.read_text(encoding="utf-8")


def test_save_audio_file_word_list_return(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineWithWordListTranscribe(), tmp_path, "user_satest3")
    async def run():
        return await session.save_audio_file()
    result = asyncio.run(run())
    assert result is not None
    txt_name = result.replace(".wav", ".txt")
    txt_path = tmp_path / "live_audio" / txt_name
    assert "bonjour whisper" in txt_path.read_text(encoding="utf-8")


def test_save_audio_file_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineWithStringTranscribe(), tmp_path, "user_satest4")
    session.full_session_audio = []
    async def run():
        return await session.save_audio_file()
    result = asyncio.run(run())
    assert result is None
