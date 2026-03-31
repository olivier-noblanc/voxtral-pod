"""
Tests de non-régression pour POST /segment_update.
Vérifie que le parsing regex fonctionne correctement sur le format
[0.10s -> 1.20s] [SPEAKER_00] texte... sans corrompre les données.
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["TESTING"] = "1"

from backend.main import app

client = TestClient(app)

_CLIENT_ID = "user_testsg"


@pytest.fixture(autouse=True)
def _transcript_file(tmp_path, monkeypatch):
    """
    Crée un fichier de transcription temporaire dans un répertoire isolé
    et redirige TRANSCRIPTIONS_DIR vers ce répertoire.
    """
    client_dir = tmp_path / _CLIENT_ID
    client_dir.mkdir(parents=True)

    transcript = client_dir / "live_20260101_000000.txt"
    transcript.write_text(
        "[0.10s -> 1.20s] [SPEAKER_00] bonjour monde\n"
        "[1.30s -> 2.40s] [SPEAKER_01] comment ça va\n",
        encoding="utf-8"
    )

    # Rediriger TRANSCRIPTIONS_DIR vers tmp_path pour l'isolation
    monkeypatch.setattr("backend.routes.api.TRANSCRIPTIONS_DIR", str(tmp_path))
    monkeypatch.setattr("backend.config.TRANSCRIPTIONS_DIR", str(tmp_path))

    return transcript


def test_segment_update_preserves_timestamp(_transcript_file):
    """Le speaker doit être modifié sans toucher au timestamp ni au texte."""
    resp = client.post("/segment_update", json={
        "client_id": _CLIENT_ID,
        "filename": "live_20260101_000000.txt",
        "segment_index": 0,
        "new_speaker": "Julie"
    })
    assert resp.status_code == 200, f"Attendu 200, obtenu {resp.status_code}: {resp.text}"
    assert resp.json().get("status") == "ok"

    content = _transcript_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Le timestamp doit être intact
    assert "[0.10s -> 1.20s]" in lines[0], f"Timestamp manquant: {lines[0]}"
    # Le nouveau speaker doit être présent
    assert "[Julie]" in lines[0], f"Nouveau speaker manquant: {lines[0]}"
    # L'ancien speaker ne doit plus être là
    assert "[SPEAKER_00]" not in lines[0], f"Ancien speaker encore présent: {lines[0]}"
    # Le texte doit être intact
    assert "bonjour monde" in lines[0], f"Texte corrompu: {lines[0]}"

    # La ligne 2 ne doit pas être touchée
    assert "[SPEAKER_01]" in lines[1], f"Ligne 2 corrompue: {lines[1]}"

def test_segment_update_triggers_profile_extraction(_transcript_file, monkeypatch):
    """Vérifier que modifier un speaker déclenche bien l'extraction de l'empreinte biométrique en tâche de fond."""
    import backend.routes.speakers as speakers_module
    
    called_args = []
    def dummy_extract_and_save(*args, **kwargs):
        called_args.append((args, kwargs))
        
    monkeypatch.setattr(speakers_module, "_extract_and_save_profile_sync", dummy_extract_and_save)

    resp = client.post("/segment_update", json={
        "client_id": _CLIENT_ID,
        "filename": "live_20260101_000000.txt",
        "segment_index": 0,
        "new_speaker": "Julie"
    })
    assert resp.status_code == 200

    assert len(called_args) == 1, "Régression: La routine d'extraction du profil n'a pas été appelée."
    args, kwargs = called_args[0]
    
    assert args[1] == 0.10, "start_s n'est pas le bon"
    assert args[2] == 1.20, "end_s n'est pas le bon"
    assert args[3] == "Julie", "Le new_speaker n'a pas été passé correctement"
    assert "live_user_testsg_20260101_000000.wav" in args[0], "Le chemin audio généré est incorrect"


def test_segment_update_second_line(_transcript_file):
    """La modification d'une ligne différente de la première doit aussi fonctionner."""
    resp = client.post("/segment_update", json={
        "client_id": _CLIENT_ID,
        "filename": "live_20260101_000000.txt",
        "segment_index": 1,
        "new_speaker": "Marc"
    })
    assert resp.status_code == 200

    content = _transcript_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    assert "[Marc]" in lines[1]
    assert "[1.30s -> 2.40s]" in lines[1]
    assert "comment ça va" in lines[1]
    # Ligne 0 intacte
    assert "[SPEAKER_00]" in lines[0]


def test_segment_update_out_of_range(_transcript_file):
    """Un segment_index hors limites doit lever 400."""
    resp = client.post("/segment_update", json={
        "client_id": _CLIENT_ID,
        "filename": "live_20260101_000000.txt",
        "segment_index": 99,
        "new_speaker": "X"
    })
    assert resp.status_code == 400


def test_segment_update_missing_client(_transcript_file):
    """Un payload incomplet doit lever 400."""
    resp = client.post("/segment_update", json={
        "filename": "live_20260101_000000.txt",
        "segment_index": 0,
        "new_speaker": "X"
    })
    assert resp.status_code == 400


def test_segment_update_file_not_found():
    """Un fichier inexistant doit lever 404."""
    resp = client.post("/segment_update", json={
        "client_id": _CLIENT_ID,
        "filename": "live_inexistant.txt",
        "segment_index": 0,
        "new_speaker": "X"
    })
    assert resp.status_code == 404
