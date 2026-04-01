"""
Tests de non-régression pour la robustesse de save_wav_only().
Vérifie que la méthode génère correctement le fichier WAV.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import pytest
from fastapi import WebSocket

from backend.core.live import LiveSession


class _WebSocketDummy:
    async def send_json(self, data: Dict[str, Any]) -> None:
        pass

def _make_audio_data() -> bytes:
    samples = np.zeros(1600, dtype=np.int16)
    return samples.tobytes()

class _EngineDummy:
    model_id = "mock"

def _build_session(engine: Any, tmp_path: Path, client_id: str = "user_satest") -> LiveSession:
    ws = _WebSocketDummy()
    session = LiveSession(engine=engine, websocket=cast(WebSocket, ws), client_id=client_id)
    session.full_session_audio = [_make_audio_data() for _ in range(5)]
    return session

@pytest.mark.asyncio
async def test_save_wav_only_creates_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineDummy(), tmp_path)
    
    wav_path, timestamp = await session.save_wav_only()
    
    assert wav_path is not None
    assert timestamp is not None
    assert os.path.exists(wav_path)
    assert wav_path.endswith(f"{timestamp}.wav")

@pytest.mark.asyncio
async def test_save_wav_only_no_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.core.live.TRANSCRIPTIONS_DIR", str(tmp_path))
    session = _build_session(_EngineDummy(), tmp_path, "user_satest4")
    session.full_session_audio = []
    
    wav_path, timestamp = await session.save_wav_only()
    
    assert wav_path is None
    assert timestamp is None
