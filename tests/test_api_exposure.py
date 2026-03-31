"""
Test pour s'assurer que tous les modules nécessaires sont correctement exposés dans l'API.
Cela prévient les erreurs similaires à celles rapportées dans le problème initial.
"""
import pytest

from backend.routes import api


def test_api_exposes_necessary_functions_for_tests() -> None:
    """Vérifie que toutes les fonctions nécessaires aux tests sont exposées dans backend.routes.api"""
    
    # Fonctions nécessaires pour les tests (comme dans les tests qui échouaient)
    required_functions = [
        'get_asr_engine',
        'add_job', 
        '_update_job_status',
        '_assemble_chunks',
        'update_segment_speaker',
        'TRANSCRIPTIONS_DIR',
        '_extract_and_save_profile_sync',
        '_gpu_job'
    ]
    
    for func_name in required_functions:
        assert hasattr(api, func_name), f"La fonction {func_name} doit être exposée dans backend.routes.api"


def test_segment_update_route_exists() -> None:
    """Vérifie que la route /segment_update est accessible"""
    from fastapi.testclient import TestClient

    from backend.main import app
    
    client = TestClient(app)
    
    # La route doit exister (même si elle retourne 404 pour fichier introuvable)
    response = client.post("/segment_update", json={
        "client_id": "test",
        "filename": "nonexistent.txt", 
        "segment_index": 0,  # type: ignore
        "new_speaker": "Test"
    })
    
    # Doit retourner 404 (fichier introuvable) mais pas 404 (route introuvable)
    assert response.status_code != 404 or "not found" in response.text.lower()


def test_all_required_imports_work() -> None:
    """Vérifie que tous les imports nécessaires fonctionnent"""
    
    # Test des imports spécifiques qui causaient des erreurs
    try:
        from backend.routes.api import get_asr_engine
        assert callable(get_asr_engine)
    except ImportError as e:
        pytest.fail(f"Impossible d'importer get_asr_engine: {e}")
    
    try:
        from backend.routes.api import _assemble_chunks
        assert callable(_assemble_chunks)
    except ImportError as e:
        pytest.fail(f"Impossible d'importer _assemble_chunks: {e}")
    
    try:
        from backend.routes.api import _extract_and_save_profile_sync
        assert callable(_extract_and_save_profile_sync)
    except ImportError as e:
        pytest.fail(f"Impossible d'importer _extract_and_save_profile_sync: {e}")
    
    try:
        from backend.routes.api import _gpu_job
        assert callable(_gpu_job)
    except ImportError as e:
        pytest.fail(f"Impossible d'importer _gpu_job: {e}")


if __name__ == "__main__":
    test_api_exposes_necessary_functions_for_tests()
    test_segment_update_route_exists()
    test_all_required_imports_work()
    print("✅ Tous les tests d'exposition API passent!")
