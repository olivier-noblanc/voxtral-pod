from typing import Any
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core.engine import TranscriptionResult

@pytest.mark.asyncio
async def test_batch_audio_persistence(tmp_path: Any) -> None:
    """Vérifie que l'audio réassemblé est bien copié dans le dossier permanent avant suppression."""
    
    # On définit les dossiers temporaires
    trans_dir = tmp_path / "transcriptions"
    os.makedirs(trans_dir / "batch_audio", exist_ok=True)
    temp_dir = tmp_path / "temp"
    os.makedirs(temp_dir / "job123", exist_ok=True)
    
    assembled_path = str(temp_dir / "job123" / "audio_full.wav")
    with open(assembled_path, "wb") as f:
        f.write(b"fake audio data")

    fake_engine = MagicMock()
    # process_file est asynchrone
    fake_engine.process_file = AsyncMock(return_value=TranscriptionResult(transcript="Dummy transcript", segments=[]))
    
    # On patche les références au moteur et les fonctions lourdes
    with patch("backend.routes.api.get_asr_engine", return_value=fake_engine), \
         patch("backend.routes.api._update_job_status"), \
         patch("backend.routes.api.add_job"), \
         patch("backend.core.postprocess.clean_text", new_callable=AsyncMock, return_value="Cleaned transcript"), \
         patch("backend.routes.api.TRANSCRIPTIONS_DIR", str(trans_dir)), \
         patch("backend.routes.api._safe_join", side_effect=lambda *args: os.path.join(*args)):
        
        from backend.routes.api import _gpu_job
        
        # Exécution du job
        await _gpu_job(assembled_path, "job123", "client456")
        
        # 1. Vérifier que la transcription (.json) est là
        client_dir = trans_dir / "client456"
        assert client_dir.exists()
        json_files = list(client_dir.glob("batch_*.json"))
        # On vérifie qu'on a le JSON principal (pas le .diar.json)
        pure_json = [f for f in json_files if not f.name.endswith(".diar.json")]
        assert len(pure_json) == 1
        
        # 2. Vérifier que l'audio est archivé dans batch_audio/
        batch_audio_dir = trans_dir / "batch_audio"
        ts = pure_json[0].name.replace("batch_", "").replace(".json", "")
        expected_audio_name = f"batch_client456_{ts}.wav"
        expected_audio_path = batch_audio_dir / expected_audio_name
        
        assert expected_audio_path.exists(), f"L'audio {expected_audio_name} n'a pas été trouvé dans {batch_audio_dir}"
        assert expected_audio_path.read_bytes() == b"fake audio data"
        
        # 3. Vérifier que le dossier temporaire est bien nettoyé
        assert not os.path.exists(assembled_path)
        assert not os.path.exists(os.path.dirname(assembled_path))
