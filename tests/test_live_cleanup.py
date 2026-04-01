import asyncio
import os
import re
import pytest
from pathlib import Path

# Import the function to test
from backend.routes.live import _live_final_job

# Helper classes to mock engine behavior
class MockResult:
    def __init__(self):
        self.transcript = "raw transcript"
        self.segments = []

    def to_rttm(self, file_id: str) -> str:
        return f"RTTM content for {file_id}"

class MockEngine:
    model_id = "mock"

    async def process_file(self, *args, **kwargs) -> MockResult:
        return MockResult()

@pytest.mark.asyncio
async def test_live_cleanup_creates_cleaned_file(tmp_path: Path, monkeypatch):
    """
    Vérifie que le nettoyage (clean_text) est bien exécuté pour les sessions live
    et qu'un fichier *.cleaned.md est créé à côté du .txt.
    """
    # ----------------------------------------------------------------------
    # Préparer l'environnement de test
    # ----------------------------------------------------------------------
    # Répertoire de transcription factice
    client_id = "user_test"
    trans_dir = tmp_path / client_id
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Patch le répertoire global utilisé par le routeur
    monkeypatch.setattr("backend.routes.live.TRANSCRIPTIONS_DIR", str(tmp_path))

    # Mock de l'engine ASR
    monkeypatch.setattr("backend.state.get_asr_engine", lambda load_model=True: MockEngine())

    # Mock des fonctions d'état de job (pas d'effet)
    monkeypatch.setattr("backend.state.add_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.state.update_job", lambda *args, **kwargs: None)

    # Mock du clean_text pour retourner un texte nettoyé connu
    async def mock_clean_text(text: str) -> str:
        return "texte nettoyé"

    monkeypatch.setattr("backend.core.postprocess.clean_text", mock_clean_text)

    # Crée un fichier wav factice (le contenu n'est pas lu)
    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"")

    # ----------------------------------------------------------------------
    # Exécuter la fonction de finalisation live
    # ----------------------------------------------------------------------
    await _live_final_job(str(wav_path), "job_test", client_id, None)

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
    # Le mock renvoie une chaîne contenant l'identifiant du fichier
    assert "RTTM content for" in rttm_content, "Le contenu du RTTM semble incorrect."
