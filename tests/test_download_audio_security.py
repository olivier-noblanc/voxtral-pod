import os
import os as _os
import shutil

# Import the FastAPI app
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from backend.config import TRANSCRIPTIONS_DIR
from backend.main import app

client = TestClient(app)

@pytest.fixture(scope="function")
def setup_test_files():
    # Create a temporary directory structure mimicking the real one
    temp_dir = tempfile.mkdtemp()
    original_dir = TRANSCRIPTIONS_DIR

    # Override the TRANSCRIPTIONS_DIR for the duration of the test
    from backend.routes import api as api_module
    api_module.TRANSCRIPTIONS_DIR = temp_dir

    client_id = "testclient"
    # Create client transcription folder with a .txt file
    client_trans_dir = os.path.join(temp_dir, client_id)
    os.makedirs(client_trans_dir, exist_ok=True)
    txt_filename = "live_20230101_120000.txt"
    txt_path = os.path.join(client_trans_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Test transcription content")

    # Create live audio file matching the naming convention
    live_audio_dir = os.path.join(temp_dir, "live_audio")
    os.makedirs(live_audio_dir, exist_ok=True)
    audio_filename = f"live_{client_id}_20230101_120000.wav"
    audio_path = os.path.join(live_audio_dir, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(b"RIFF....WAVE")  # minimal placeholder header

    yield {
        "client_id": client_id,
        "audio_filename": audio_filename,
        "temp_dir": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir)
    # Restore original constant
    api_module.TRANSCRIPTIONS_DIR = original_dir

def test_download_audio_unauthorized(setup_test_files):
    data = setup_test_files
    # Use a different client_id to simulate unauthorized access
    wrong_client_id = "otherclient"
    resp = client.get(f"/download_audio/{wrong_client_id}/{data['audio_filename']}")
    assert resp.status_code == 403
    # FastAPI returns JSON error detail
    json_body = resp.json()
    assert "detail" in json_body
    assert "Accès refusé" in json_body["detail"]