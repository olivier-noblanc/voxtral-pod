"""
Tests pour la suppression de l'historique des transcriptions et des fichiers audio associés.
"""
import os

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app instance
from backend.main import app

client = TestClient(app)


@pytest.fixture
def setup_transcription():
    """
    Crée un fichier de transcription et son fichier audio associé
    dans le répertoire de test, puis le nettoie après le test.
    """
    client_id = "user_test"
    # Répertoire de transcription
    trans_dir = os.path.join("transcriptions_terminees", client_id)
    os.makedirs(trans_dir, exist_ok=True)

    # Fichier de transcription (exemple live)
    filename = "live_test.txt"
    trans_path = os.path.join(trans_dir, filename)
    with open(trans_path, "w", encoding="utf-8") as f:
        f.write("Contenu de test")

    # Fichier audio associé (live_audio)
    audio_dir = os.path.join("transcriptions_terminees", "live_audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_name = f"live_{client_id}_test.wav"
    audio_path = os.path.join(audio_dir, audio_name)
    with open(audio_path, "wb") as f:
        f.write(b"")

    yield {
        "client_id": client_id,
        "filename": filename,
        "trans_path": trans_path,
        "audio_path": audio_path,
    }

    # Nettoyage après le test
    for path in (trans_path, audio_path):
        if os.path.isfile(path):
            os.remove(path)


def test_delete_transcription_endpoint(setup_transcription):
    """
    Vérifie que l'endpoint DELETE supprime correctement le fichier de transcription
    ainsi que le fichier audio associé et renvoie le bon statut.
    """
    # On utilise l'argument setup_transcription qui contient les données de la fixture
    trans_data = setup_transcription
    client_id = trans_data["client_id"]
    filename = trans_data["filename"]
    trans_path = trans_data["trans_path"]
    audio_path = trans_data["audio_path"]

    # L'endpoint doit répondre 200 avec le corps {"status": "ok"}
    response = client.delete(f"/transcriptions/{filename}?client_id={client_id}")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Les fichiers doivent être supprimés du système de fichiers
    assert not os.path.isfile(trans_path), "Le fichier de transcription n'a pas été supprimé"
    assert not os.path.isfile(audio_path), "Le fichier audio associé n'a pas été supprimé"
