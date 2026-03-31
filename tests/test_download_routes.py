import os
import shutil
import tempfile

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient

# Import the FastAPI app (adjust sys.path for relative import)
import os as _os
import sys

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

    # Create live audio file matching the naming convention used by LiveSession.save_audio_file
    live_audio_dir = os.path.join(temp_dir, "live_audio")
    os.makedirs(live_audio_dir, exist_ok=True)
    audio_filename = f"live_{client_id}_20230101_120000.wav"
    audio_path = os.path.join(live_audio_dir, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(b"RIFF....WAVE")  # minimal placeholder header

    # Create batch audio file
    batch_audio_dir = os.path.join(temp_dir, "batch_audio")
    os.makedirs(batch_audio_dir, exist_ok=True)
    batch_txt = "batch_20230101_130000.txt"
    batch_audio = f"batch_{client_id}_20230101_130000.wav"
    with open(os.path.join(batch_audio_dir, batch_audio), "wb") as f:
        f.write(b"RIFF....WAVE")

    yield {
        "client_id": client_id,
        "txt_filename": txt_filename,
        "audio_filename": audio_filename,
        "batch_txt": batch_txt,
        "batch_audio": batch_audio,
        "temp_dir": temp_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir)
    # Restore original constant
    api_module.TRANSCRIPTIONS_DIR = original_dir

def test_download_live_audio(setup_test_files):
    data = setup_test_files
    resp = client.get(f"/download_audio/{data['client_id']}/{data['audio_filename']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # Simple sanity check on content
    assert resp.content.startswith(b"RIFF")

def test_download_batch_audio(setup_test_files):
    data = setup_test_files
    resp = client.get(f"/download_audio/{data['client_id']}/{data['batch_audio']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"

def test_download_transcript(setup_test_files):
    data = setup_test_files
    resp = client.get(f"/download_transcript/{data['client_id']}/{data['txt_filename']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain"
    assert b"Test transcription content" in resp.content
