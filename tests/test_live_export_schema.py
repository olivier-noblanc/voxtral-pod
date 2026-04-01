import pytest
from pathlib import Path

# Import the function to test
from backend.routes.live import _live_final_job

# Mock engine that returns a result with a malformed segment (missing ``start``)
class BadResult:
    transcript = "texte brut"
    # ``end`` present but ``start`` absent → should trigger validation error
    segments = [{"end": 2.5, "text": "segment sans start"}]

    def to_rttm(self, file_id: str) -> str:
        return "RTTM placeholder"

class BadEngine:
    model_id = "mock"

    async def process_file(self, *args, **kwargs) -> BadResult:
        return BadResult()

@pytest.mark.asyncio
async def test_live_export_schema_validation(tmp_path: Path, monkeypatch):
    """
    Le pipeline doit lever ``ValueError`` lorsqu'un segment ne possède pas
    les champs ``start`` et ``end``.  Aucun fichier texte/RTTM ne doit être
    créé dans ce cas.
    """
    client_id = "user_test"
    # Le répertoire de transcription factice
    trans_dir = tmp_path / client_id
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Patch le répertoire global utilisé par le routeur
    monkeypatch.setattr("backend.routes.live.TRANSCRIPTIONS_DIR", str(tmp_path))

    # Mock de l'engine ASR
    monkeypatch.setattr("backend.state.get_asr_engine", lambda load_model=True: BadEngine())

    # Mock des fonctions d'état de job (pas d'effet)
    monkeypatch.setattr("backend.state.add_job", lambda *args, **kwargs: None)
    monkeypatch.setattr("backend.state.update_job", lambda *args, **kwargs: None)

    # Le nettoyage n'est pas pertinent pour ce test
    monkeypatch.setattr("backend.core.postprocess.clean_text", lambda text: "texte nettoyé")

    wav_path = tmp_path / "dummy.wav"
    wav_path.write_bytes(b"")

    # Le pipeline doit lever une exception de validation
    with pytest.raises(ValueError, match="missing required timestamps"):
        await _live_final_job(str(wav_path), "job_test", client_id, None)

    # Aucun fichier exporté ne doit exister
    exported = list(trans_dir.glob("*"))
    assert exported == [], "Aucun fichier (TXT/RTTM/JSON) ne doit être créé en cas d'erreur de schéma."