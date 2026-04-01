import asyncio
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import soundfile as sf


# Helper classes to mock engine behavior
class MockResult:
    def __init__(self) -> None:
        self.transcript = "raw transcript"
        self.segments: List[Dict[str, Any]] = []

class MockEngine:
    model_id = "mock"

    async def process_file(self, *args: Any, **kwargs: Any) -> MockResult:
        return MockResult()

def test_live_cleanup_creates_cleaned_file(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Vérifie que le nettoyage (clean_text) est bien exécuté pour les sessions live
    et qu'un fichier *.cleaned.md est créé à côté du .json.
    
    Note: Puisque asyncio_mode=strict dans pytest.ini, ce test n'est pas géré par le 
    Runner de pytest-asyncio. On peut donc utiliser asyncio.run() en isolation totale.
    """
    # Import locally
    from backend.routes.live import _live_final_job

    client_id = "user_test"
    trans_dir = tmp_path / client_id
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Patch le répertoire global utilisé par le routeur
    monkeypatch.setattr("backend.routes.live.TRANSCRIPTIONS_DIR", str(tmp_path))

    # Mock du clean_text pour retourner un texte nettoyé connu
    async def mock_clean_text(text: str) -> str:
        return "texte nettoyé"
    
    # Mock de l'export RTTM
    def mock_export_rttm(segments: Any, file_id: str, rttm_path: str | Path) -> None:
        with open(rttm_path, "w", encoding="utf-8") as f:
            f.write(f"RTTM content for {file_id}")

    # Mock de l'engine ASR et des functions d'état directement dans le namespace de live.py
    monkeypatch.setattr("backend.routes.live.get_asr_engine", lambda load_model=True: MockEngine())
    monkeypatch.setattr("backend.routes.live.add_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.routes.live.update_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.routes.live.clean_text", mock_clean_text)
    
    monkeypatch.setattr("backend.core.export.export_rttm", mock_export_rttm)
    monkeypatch.setattr("backend.core.export.validate_segments", lambda x: None)

    # Crée un fichier wav valid (silence)
    wav_path = tmp_path / "dummy.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype='float32'), 16000)

    # ----------------------------------------------------------------------
    # Exécuter la function de finalisation live à l'intérieur d'asyncio.run
    # ----------------------------------------------------------------------
    asyncio.run(_live_final_job(str(wav_path), "job_test", client_id, None))

    # ----------------------------------------------------------------------
    # Vérifier la présence du fichier *.cleaned.md
    # ----------------------------------------------------------------------
    cleaned_files = list(trans_dir.glob("*.cleaned.md"))
    assert len(cleaned_files) == 1, "Le fichier de nettoyage n'a pas été créé."

    cleaned_content = cleaned_files[0].read_text(encoding="utf-8")
    assert cleaned_content == "texte nettoyé", "Le contenu du fichier nettoyé est incorrect."

    # ----------------------------------------------------------------------
    # Vérifier que le fichier RTTM a bien été créé et n'a pas été altéré
    # ----------------------------------------------------------------------
    rttm_files = list(trans_dir.glob("*.rttm"))
    assert len(rttm_files) == 1, "Le fichier RTTM n'a pas été créé."

    rttm_content = rttm_files[0].read_text(encoding="utf-8")
    assert "RTTM content for" in rttm_content, "Le contenu du RTTM semble incorrect."
