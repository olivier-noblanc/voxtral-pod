import os
import shutil
import tempfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.main import app

client = TestClient(app)

@pytest.fixture
def mock_postprocess():
    with patch("backend.core.postprocess.summarize_text", return_value="### Summary\n- Point A\n- Point B") as m1, \
         patch("backend.core.postprocess.extract_actions_text", return_value=["Action 1", "Action 2"]) as m2, \
         patch("backend.core.postprocess.clean_text", return_value="**Cleaned** Content") as m3:
        yield (m1, m2, m3)

@pytest.fixture
def setup_data():
    temp_dir = tempfile.mkdtemp()
    from backend.routes import api as api_module
    from backend import config as config_module
    old_dir = api_module.TRANSCRIPTIONS_DIR
    api_module.TRANSCRIPTIONS_DIR = temp_dir
    # Also update the config constant used elsewhere
    config_module.TRANSCRIPTIONS_DIR = temp_dir
    
    client_id = "user_test123"
    client_dir = os.path.join(temp_dir, client_id)
    os.makedirs(client_dir)
    filename = "test.txt"
    with open(os.path.join(client_dir, filename), "w", encoding="utf-8") as f:
        f.write("Some transcription text")
        
    yield {"client_id": client_id, "filename": filename, "temp_dir": temp_dir}
    
    shutil.rmtree(temp_dir)
    api_module.TRANSCRIPTIONS_DIR = old_dir
    config_module.TRANSCRIPTIONS_DIR = old_dir

def test_view_summary_route(setup_data, mock_postprocess):
    data = setup_data
    resp = client.get(f"/view_summary/{data['client_id']}/{data['filename']}")
    assert resp.status_code == 200
    assert "Compte Rendu Albert" in resp.text
    assert "marked.min.js" not in resp.text
    assert "postprocess.js" in resp.text
    assert "postprocess.css" in resp.text
    # Server-side rendering verification: h3 and ul tags should be present
    assert "<h3>Summary</h3>" in resp.text
    assert "<ul>" in resp.text
    assert "<li>Point A</li>" in resp.text

def test_view_actions_route(setup_data, mock_postprocess):
    data = setup_data
    resp = client.get(f"/view_actions/{data['client_id']}/{data['filename']}")
    assert resp.status_code == 200
    assert "Actions Albert" in resp.text
    assert "<li>Action 1</li>" in resp.text
    assert "data-content=" in resp.text

def test_view_cleanup_route(setup_data, mock_postprocess):
    data = setup_data
    resp = client.get(f"/view_cleanup/{data['client_id']}/{data['filename']}")
    assert resp.status_code == 200
    assert "Nettoyage Albert" in resp.text
    assert "<strong>Cleaned</strong>" in resp.text

def test_view_summary_isolation(setup_data, mock_postprocess):
    # Try to access as another client
    data = setup_data
    resp = client.get(f"/view_summary/user_other456/{data['filename']}")
    # _safe_join or _validate_client_id should catch this or return 404 since dir doesn't exist
    assert resp.status_code in [400, 404]
