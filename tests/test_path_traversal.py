from typing import Any
"""
test_path_traversal.py
----------------------
Vérifie que les routes HTTP rejettent les tentatives de path traversal
(client_id ou filename contenant des séquences comme ../../).

Régresse : Bug #3 — absence de validation de chemin dans routes/api.py.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)

# Payloads classiques de path traversal
TRAVERSALS = [
    "../etc/passwd",
    "../../etc/passwd",
    "..%2Fetc%2Fpasswd",
    "..%252F..%252Fetc%252Fpasswd",
    "/etc/passwd",
    "....//....//etc/passwd",
]


@pytest.mark.parametrize("payload", TRAVERSALS)
def test_list_trans_path_traversal(payload):
    """/transcriptions?client_id=<payload> doit retourner 400 ou [] jamais 200 avec un contenu."""
    resp = client.get("/transcriptions", params={"client_id": payload})
    # Soit rejeté (400), soit dossier introuvable (retourne [])
    assert resp.status_code in (400, 200), f"Unexpected status {resp.status_code}"
    if resp.status_code == 200:
        assert resp.json() == [], "path traversal should return empty list, not real content"


@pytest.mark.parametrize("payload", TRAVERSALS)
def test_get_trans_path_traversal(payload):
    """/transcription/{filename}?client_id=<payload> doit retourner 400 ou 404."""
    resp = client.get("/transcription/test.txt", params={"client_id": payload})
    assert resp.status_code in (400, 404), (
        f"path traversal via client_id should be blocked, got {resp.status_code}"
    )


@pytest.mark.parametrize("payload", TRAVERSALS)
def test_download_transcript_path_traversal_client(payload):
    """/download_transcript/{client_id}/file doit retourner 400 ou 404."""
    resp = client.get(f"/download_transcript/{payload}/test.txt")
    assert resp.status_code in (400, 404), (
        f"path traversal via client_id should be blocked, got {resp.status_code}"
    )


@pytest.mark.parametrize("payload", TRAVERSALS)
def test_download_transcript_path_traversal_filename(payload):
    """/download_transcript/client/{filename} doit retourner 400 ou 404."""
    resp = client.get(f"/download_transcript/client/{payload}")
    assert resp.status_code in (400, 404), (
        f"path traversal via filename should be blocked, got {resp.status_code}"
    )


@pytest.mark.parametrize("payload", TRAVERSALS)
def test_view_path_traversal(payload):
    """/view/{client_id}/{filename} doit retourner 400 ou 404."""
    resp = client.get(f"/view/{payload}/test.txt")
    assert resp.status_code in (400, 404), (
        f"path traversal via client_id should be blocked, got {resp.status_code}"
    )


# ── Test unitaire direct sur _safe_join ─────────────────────────────────────

def test_safe_join_raises_on_traversal(tmp_path: Any) -> None:
    """_safe_join doit lever HTTPException 400 sur les chemins malveillants."""
    from fastapi import HTTPException

    from backend.routes.api import _safe_join

    base = str(tmp_path)

    # Chemin légitime → pas d'exception
    result = _safe_join(base, "client1", "file.txt")
    assert result.startswith(base)

    # Chemin traversant → 400
    with pytest.raises(HTTPException) as exc_info:
        _safe_join(base, "../../etc/passwd")
    assert exc_info.value.status_code == 400


def test_safe_join_legitimate_paths(tmp_path: Any) -> None:
    """_safe_join laisse passer les chemins normaux."""
    from backend.routes.api import _safe_join

    base = str(tmp_path)
    for parts in [
        ("client_abc", "batch_20240101.txt"),
        ("user-123", "live_20240101_120000.txt"),
    ]:
        result = _safe_join(base, *parts)
        assert result.startswith(base)
