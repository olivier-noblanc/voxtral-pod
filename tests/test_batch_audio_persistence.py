import os
import shutil
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from backend.core.engine import TranscriptionResult

def test_batch_audio_persistence(tmp_path):
    """Vérifie que l'audio réassemblé est bien copié dans le dossier permanent avant suppression."""
    
    # On définit les dossiers temporaires
    trans_dir = tmp_path / "transcriptions"
    os.makedirs(trans_dir / "batch_audio", exist_ok=True)
    temp_dir = tmp_path / "temp"
    os.makedirs(temp_dir / "job123", exist_ok=True)
    
    assembled_path = str(temp_dir / "job123" / "audio_full.wav")
    with open(assembled_path, "wb") as f:
        f.write(b"fake audio data")

    async def _pf_mock(*args, **kwargs):
        return TranscriptionResult(transcript="Dummy transcript", segments=[])

    fake_engine = MagicMock()
    fake_engine.process_file.side_effect = _pf_mock
    
    # On patche les références au moteur et les fonctions lourdes
    with patch("backend.routes.api.get_asr_engine", return_value=fake_engine), \
         patch("backend.routes.api._update_job_status"), \
         patch("backend.routes.api.add_job"), \
         patch("backend.core.postprocess.clean_text", new_callable=AsyncMock, return_value="Cleaned transcript"), \
         patch("backend.routes.api.TRANSCRIPTIONS_DIR", str(trans_dir)), \
         patch("backend.routes.api._safe_join", side_effect=lambda *args: os.path.join(*args)):
        
        from backend.routes.api import _gpu_job
        
        async def run_test():
            # Exécution du job
            await _gpu_job(assembled_path, "job123", "client456")
            
            # 1. Vérifier que la transcription (.txt) est là
            client_dir = trans_dir / "client456"
            assert client_dir.exists()
            txt_files = list(client_dir.glob("batch_*.txt"))
            assert len(txt_files) == 1
            
            # 2. Vérifier que l'audio est archivé dans batch_audio/
            batch_audio_dir = trans_dir / "batch_audio"
            ts = txt_files[0].name.replace("batch_", "").replace(".txt", "")
            expected_audio_name = f"batch_client456_{ts}.wav"
            expected_audio_path = batch_audio_dir / expected_audio_name
            
            assert expected_audio_path.exists(), f"L'audio {expected_audio_name} n'a pas été trouvé dans {batch_audio_dir}"
            assert expected_audio_path.read_bytes() == b"fake audio data"
            
            # 3. Vérifier que le dossier temporaire est bien nettoyé
            assert not os.path.exists(assembled_path)
            assert not os.path.exists(os.path.dirname(assembled_path))

        asyncio.run(run_test())
