import os
import json
import subprocess
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app defined in backend/main.py
from backend.main import app

client = TestClient(app)

def test_get_speaker_profiles():
    """La route /speaker_profiles doit renvoyer un code 200 et une liste (vide ou non)."""
    response = client.get("/speaker_profiles")
    assert response.status_code == 200, "Le code de statut doit être 200"
    data = response.json()
    assert isinstance(data, list), "La réponse doit être une liste"
    # La liste peut être vide, mais le type doit être correct
    for item in data:
        assert isinstance(item, dict), "Chaque élément de la liste doit être un dict"
        assert "speaker_id" in item and "name" in item, "Chaque dict doit contenir speaker_id et name"

def test_status_route_returns_git_info():
    """La route /status/{file_id} doit retourner les informations système et la version Git si disponible."""
    # Utiliser un identifiant de fichier factice
    response = client.get("/status/dummy_id")
    assert response.status_code == 200, "Le code de statut doit être 200"
    data = response.json()
    # Le statut doit contenir au moins la clé 'status'
    assert "status" in data, "Le JSON doit contenir la clé 'status'"

    # Vérifier la présence du commit Git si le dépôt est accessible
    # On tente de récupérer le commit via git, sinon on accepte l'absence
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), text=True
        ).strip()[:7]
        # Si le commit a pu être récupéré, il doit être présent dans la réponse
        assert data.get("commit") == commit, "Le commit Git doit correspondre à celui du dépôt"
    except Exception:
        # Si git n'est pas disponible, on accepte simplement que la clé 'commit' soit absente ou égale à 'unknown'
        assert data.get("commit") in (None, "unknown")

def test_configuration_flags_are_respected():
    """Vérifier que les paramètres de configuration comme allow_tf32 sont bien pris en compte."""
    # Avant d'appeler un endpoint qui initialise le moteur, on s'assure que le flag est désactivé (par défaut)
    import torch
    # Le flag peut déjà être True si le GPU est disponible, on le réinitialise pour le test
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Appeler un endpoint qui déclenche l'initialisation du moteur (ex: /status)
    # Cela passe par get_asr_engine qui appelle setup_gpu dans backend/config
    _ = client.get("/status/dummy_id")

    # Après l'appel, le flag doit être activé si CUDA est disponible
    if torch.cuda.is_available():
        assert torch.backends.cuda.matmul.allow_tf32 is True, "allow_tf32 doit être activé sur CUDA"
        assert torch.backends.cudnn.allow_tf32 is True, "allow_tf32 doit être activé sur cuDNN"
    else:
        # En l'absence de GPU, le flag reste False mais aucune exception ne doit être levée
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False