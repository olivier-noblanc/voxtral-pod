"""
Tests de non-régression pour la robustesse de save_wav_only().
Vérifie que la méthode génère correctement le fichier WAV.
"""
import asyncio
import os
import numpy as np

from backend.core.live import LiveSession

class _WebSocketDummy:
    async def send_json(self, data): pass

def _make_audio_data():
    samples = np.zeros(1600, dtype=np.int16)
    return samples.tobytes()

class _EngineDummy:
    model_id = "mock"

def _build_session(engine, tmp_path, client_id="user_satest") -> LiveSession:
    from typing import cast
    from fastapi import WebSocket
    ws = _WebSocketDummy()
    session = LiveSession(engine=engine, websocket=cast(WebSocket, ws), client_id=client_id)
    session.full_session_audio = [_make_audio_data() for _ in range(5)]
    return session

def test_save_wav_only_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineDummy(), tmp_path)
    async def run():
        return await session.save_wav_only()
    wav_path, timestamp = asyncio.run(run())
    
    assert wav_path is not None
    assert timestamp is not None
    assert os.path.exists(wav_path)
    assert wav_path.endswith(f"{timestamp}.wav")

def test_save_wav_only_no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineDummy(), tmp_path, "user_satest4")
    session.full_session_audio = []
    async def run():
        return await session.save_wav_only()
    wav_path, timestamp = asyncio.run(run())
    assert wav_path is None
    assert timestamp is None
