from fastapi import HTTPException

from backend.config import TRANSCRIPTIONS_DIR
from backend.routes.api import _safe_join


def test_safe_join_traversal() -> None:
    """Vérifie que _safe_join lève une exception en cas de path traversal."""
    # Chemin valide
    try:
        path = _safe_join(TRANSCRIPTIONS_DIR, "live_audio", "test.wav")
        assert "live_audio" in path
    except HTTPException:
        assert False, "_safe_join devrait accepter un chemin valide"

    # Tentative de sortie du répertoire de base
    try:
        _safe_join(TRANSCRIPTIONS_DIR, "live_audio", "../../../etc/passwd")
        assert False, "_safe_join devrait bloquer le path traversal"
    except HTTPException as e:
        assert e.status_code == 400
        assert e.detail == "Chemin invalide."

if __name__ == "__main__":
    print("[*] Running Security Audit Tests...")
    test_safe_join_traversal()
    print("[+] Security Audit Tests Passed!")
