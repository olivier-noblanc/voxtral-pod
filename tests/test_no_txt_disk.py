from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.routes.api import TRANSCRIPTIONS_DIR


@pytest.mark.asyncio
async def test_batch_transcription_no_txt_on_disk(client: TestClient) -> None:
    """
    Simule une transcription batch et vérifie qu'aucun fichier .txt n'est créé.
    """
    client_id = "test_no_txt_user"
    from backend.core.engine import TranscriptionResult
    from backend.routes.audio import _gpu_job
    
    mock_result = TranscriptionResult(
        transcript="Test sans txt",
        segments=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Test"}]
    )
    
    # On mocke l'engine et shutil pour éviter les accès disque inutiles
    with patch("backend.routes.api.get_asr_engine") as mock_get_engine, \
         patch("shutil.copy2") as mock_copy, \
         patch("os.remove") as mock_remove:
         
        mock_engine = MagicMock()
        mock_engine.process_file = AsyncMock(return_value=mock_result)
        mock_get_engine.return_value = mock_engine
        
        # On exécute le job
        try:
            await _gpu_job("mock_audio.wav", "job_001", client_id)
        except Exception as e:
            pytest.fail(f"_gpu_job a échoué : {e}")
    
    user_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    
    # On ne connaît pas le timestamp exact car il est généré dans _gpu_job
    # Mais on peut lister le contenu du dossier
    if not os.path.exists(user_dir):
        pytest.fail(f"Le dossier utilisateur {user_dir} n'a pas été créé.")
        
    files = os.listdir(user_dir)
    has_json = any(f.endswith(".json") for f in files)
    has_txt = any(f.endswith(".txt") for f in files)
    
    try:
        assert has_json, f"Le fichier JSON devrait être créé. Fichiers: {files}"
        assert not has_txt, f"ERREUR : Un fichier .txt a été créé sur le disque ! Fichiers: {files}"
    finally:
        # Cleanup
        import shutil
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)


def test_list_trans_uses_json(client: TestClient) -> None:
    """
    Vérifie que la liste des transcriptions se base sur les fichiers .json.
    """
    client_id = "test_list_json_user"
    user_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(user_dir, exist_ok=True)
    
    json_file = "batch_20240101_120000.json"
    json_path = os.path.join(user_dir, json_file)
    
    # Créer un fichier JSON
    with open(json_path, "w") as f:
        json.dump([{"text": "ok"}], f)
        
    try:
        # Appeler l'API
        response = client.get(f"/transcriptions?client_id={client_id}")
        assert response.status_code == 200
        data = response.json()
        
        # On s'attend à voir le fichier
        assert any("20240101_120000" in item["name"] for item in data)
    finally:
        if os.path.exists(json_path): os.remove(json_path)
        if os.path.exists(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)
