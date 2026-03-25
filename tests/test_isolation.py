import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.main import app

client = TestClient(app)

@pytest.fixture
def setup_isolation_env():
    # Setup temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Path mocking
    import backend.routes.api as api_mod
    import backend.config as config_mod
    orig_dir = config_mod.TRANSCRIPTIONS_DIR
    api_mod.TRANSCRIPTIONS_DIR = temp_dir
    config_mod.TRANSCRIPTIONS_DIR = temp_dir

    # User A data
    user_a = "user_aaaa"
    user_a_dir = os.path.join(temp_dir, user_a)
    os.makedirs(user_a_dir, exist_ok=True)
    with open(os.path.join(user_a_dir, "file_a.txt"), "w") as f: f.write("content a")
    
    # User A audio
    batch_audio_dir = os.path.join(temp_dir, "batch_audio")
    os.makedirs(batch_audio_dir, exist_ok=True)
    audio_a = f"batch_{user_a}_123.wav"
    with open(os.path.join(batch_audio_dir, audio_a), "wb") as f: f.write(b"audio a content")
    
    # User A live temp
    live_audio_dir = os.path.join(temp_dir, "live_audio")
    os.makedirs(live_audio_dir, exist_ok=True)
    live_txt_a = f"live_{user_a}_456.txt"
    with open(os.path.join(live_audio_dir, live_txt_a), "w") as f: f.write("live content a")

    yield {
        "user_a": user_a,
        "user_b": "user_bbbb",
        "file_a": "file_a.txt",
        "audio_a": audio_a,
        "live_txt_a": live_txt_a
    }

    # Cleanup
    shutil.rmtree(temp_dir)
    api_mod.TRANSCRIPTIONS_DIR = orig_dir
    config_mod.TRANSCRIPTIONS_DIR = orig_dir

def test_isolation_list(setup_isolation_env):
    env = setup_isolation_env
    # User B should see nothing
    resp = client.get(f"/transcriptions?client_id={env['user_b']}")
    assert resp.status_code == 200
    assert resp.json() == []

def test_isolation_download_transcript(setup_isolation_env):
    env = setup_isolation_env
    # User B should be blocked from downloading User A's transcript
    # (By route logic: file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename))
    # Since user_b directory doesn't exist, it should be 404/400
    resp = client.get(f"/download_transcript/{env['user_b']}/{env['file_a']}")
    assert resp.status_code in [404, 400]

def test_isolation_download_audio(setup_isolation_env):
    env = setup_isolation_env
    # User B tries to download User A's audio by providing their own client_id in URL
    resp = client.get(f"/download_audio/{env['user_b']}/{env['audio_a']}")
    assert resp.status_code == 403
    assert "Accès refusé" in resp.json()["detail"]

def test_isolation_view_live_temp(setup_isolation_env):
    env = setup_isolation_env
    # User B tries to view User A's temporary live transcript
    resp = client.get(f"/view/{env['user_b']}/{env['live_txt_a']}")
    assert resp.status_code == 403

def test_websocket_forbidden(setup_isolation_env):
    # Try connecting with anonymous or invalid ID
    with pytest.raises(Exception): # FastAPI TestClient raises if closed immediately
        with client.websocket_connect("/live?client_id=anonymous") as websocket:
            pass

def test_batch_chunk_forbidden(setup_isolation_env):
    # Try batch chunk with anonymous
    resp = client.post("/batch_chunk", data={
        "file_id": "job123",
        "client_id": "anonymous",
        "chunk_index": 0,
        "total_chunks": 1
    }, files={"file": ("test.wav", b"data")})
    assert resp.status_code == 400
