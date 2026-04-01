from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.routes.live import _live_final_job
from backend.state import get_job


# Mock engine that returns a result with a malformed segment (missing ``start``)
class BadResult:
    transcript: str = "texte brut"
    # ``end`` present but ``start`` absent → should trigger validation error
    segments: List[Dict[str, Any]] = [{"end": 2.5, "text": "segment sans start"}]

    def to_rttm(self, file_id: str) -> str:
        return f"SPEAKER {file_id} 1 0.000 0.000 <NA> <NA> mock <NA> <NA>"


class BadEngine:
    model_id: str = "mock"

    async def process_file(self, *args: Any, **kwargs: Any) -> BadResult:
        """Mock process_file documentation."""
        return BadResult()


def generate_silent_wav(path: Path) -> None:
    """Génère un fichier WAV valide (silence 1s) sans dépendances externes."""
    sample_rate = 16000
    num_samples = 16000
    # Header minimal pour PCM 16-bit Mono 16kHz
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + num_samples * 2, b'WAVE', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b'data',
        num_samples * 2
    )
    samples = b'\x00\x00' * num_samples
    path.write_bytes(header + samples)


@pytest.mark.asyncio
async def test_live_export_schema_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Le pipeline ne doit pas lever d'exception (background job) mais doit marquer 
    le job en 'erreur' lorsqu'un segment ne possède pas les champs 'start'/'end'.
    Aucun fichier TXT/RTTM/JSON ne doit être créé sur le disque.
    """
    client_id = "user_test"
    job_id = "job_validation_test"

    # Le répertoire de transcription factice
    trans_dir = tmp_path / client_id
    trans_dir.mkdir(parents=True, exist_ok=True)

    # Patch le répertoire global utilisé par le routeur
    monkeypatch.setattr("backend.routes.live.TRANSCRIPTIONS_DIR", str(tmp_path))

    # Mock de l'engine ASR (patché là où il est utilisé)
    monkeypatch.setattr("backend.routes.live.get_asr_engine", lambda load_model=True: BadEngine())

    # On utilise les vraies fonctions d'état (DB locale via conftest)
    # Monkeypatch de clean_text pour éviter des appels API externes
    monkeypatch.setattr("backend.routes.live.clean_text", lambda text: "texte nettoyé")

    # Bug 1 Fix : Génère un VRAI fichier WAV valide pour éviter le crash ffmpeg
    wav_path = tmp_path / "valid_silence.wav"
    generate_silent_wav(wav_path)

    # Exécution du job (ne doit pas lever d'exception grâce au try/except global)
    await _live_final_job(str(wav_path), job_id, client_id, None)

    # Vérification dans la base de données
    job_data = get_job(job_id)
    assert job_data.get("status") == "erreur", "Le job devrait être marqué en 'erreur' suite à l'échec de validation."
    error_details: str = job_data.get("error_details", "")
    assert "missing required timestamps" in error_details, "Le détail de l'erreur doit être présent."

    # Bug 2 Fix : Aucun fichier exporté ne doit exister (JSON/RTTM supprimés ou jamais écrits)
    exported = list(trans_dir.glob("*"))
    assert exported == [], f"Aucun fichier (JSON/RTTM) ne devrait être présent. Trouvé : {exported}"