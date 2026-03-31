"""
Tests de non-régression pour l'authentification admin.
Vérifie que /git_update et /change_model nécessitent X-Admin-Key
quand ADMIN_API_KEY est définie dans l'environnement.
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["TESTING"] = "1"

from backend.main import app

client = TestClient(app)

_VALID_KEY = "test_admin_secret"


@pytest.fixture(autouse=True)
def mock_heavy_operations(monkeypatch):
    """Bouche les appels lourds pour éviter un vrai git reset ou le chargement des modèles."""
    def mock_run(*args, **kwargs):
        class CompletedProc:
            stdout = "Mocked update"
            stderr = ""
        return CompletedProc()
    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr("backend.routes.api.get_asr_engine", lambda load_model=False: None)


class TestChangeModelAuth:
    """Tests pour POST /change_model avec et sans ADMIN_API_KEY."""

    def test_change_model_no_env_key_is_accessible(self, monkeypatch):
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        resp = client.post("/api/change_model?model=mock")
        assert resp.status_code == 200, f"Attendu 200 sans clé env, obtenu {resp.status_code}"

    def test_change_model_wrong_key_returns_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", _VALID_KEY)
        resp = client.post("/api/change_model?model=mock", headers={"X-Admin-Key": "mauvaise"})
        assert resp.status_code == 403

    def test_change_model_no_key_header_returns_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", _VALID_KEY)
        resp = client.post("/api/change_model?model=mock")
        assert resp.status_code == 403

    def test_change_model_valid_key_returns_200(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", _VALID_KEY)
        resp = client.post("/api/change_model?model=mock", headers={"X-Admin-Key": _VALID_KEY})
        assert resp.status_code == 200
        assert resp.json().get("status") == "ok"


class TestGitUpdateAuth:
    """Tests pour POST /git_update avec et sans ADMIN_API_KEY."""

    def test_git_update_no_env_key_does_not_raise_403(self, monkeypatch):
        monkeypatch.delenv("ADMIN_API_KEY", raising=False)
        resp = client.post("/api/git_update")
        assert resp.status_code != 403

    def test_git_update_wrong_key_returns_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", _VALID_KEY)
        resp = client.post("/api/git_update", headers={"X-Admin-Key": "wrong"})
        assert resp.status_code == 403

    def test_git_update_missing_header_returns_403(self, monkeypatch):
        monkeypatch.setenv("ADMIN_API_KEY", _VALID_KEY)
        resp = client.post("/api/git_update")
        assert resp.status_code == 403
