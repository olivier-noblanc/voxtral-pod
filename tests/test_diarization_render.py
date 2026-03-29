import pytest
from fastapi.templating import Jinja2Templates
from unittest.mock import MagicMock
import os

# Configurer le chemin des templates (relatif à la racine du projet)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "backend", "templates")

@pytest.fixture
def templates():
    return Jinja2Templates(directory=TEMPLATES_DIR)

def test_diarization_view_rendering(templates):
    """
    Vérifie que le template diarization_view.html est rendu sans erreur Jinja2.
    """
    mock_request = MagicMock()
    
    # Données représentatives de ce que renvoie build_speaker_segments
    segments = [
        {"speaker": "LOCUTEUR_A", "start": 0.0, "end": 10.5, "text": "Ceci est un test."},
        {"speaker": "LOCUTEUR_B", "start": 10.5, "end": 20.0, "text": "Deuxième segment."},
    ]
    
    # Simuler le passage de segments bruts (sans duration calculée à l'avance)
    context = {
        "request": mock_request,
        "segments": segments,
        "filename": "audio_test.wav",
        "client_id": "user_mock_123"
    }
    
    # On utilise get_template().render() pour tester le moteur Jinja2 en profondeur
    template = templates.get_template("diarization_view.html")
    rendered = template.render(context)
    
    assert "Diarisation" in rendered
    assert "LOCUTEUR_A" in rendered
    assert "10.500s" in rendered # Test du filtre format %.3f
    assert "audio_test.wav" in rendered

def test_diarization_view_rendering_empty(templates):
    """
    Vérifie le rendu avec une liste de segments vide.
    """
    mock_request = MagicMock()
    context = {
        "request": mock_request,
        "segments": [],
        "filename": "empty.wav",
        "client_id": "user_mock_123"
    }
    template = templates.get_template("diarization_view.html")
    rendered = template.render(context)
    assert "Diarisation" in rendered
    assert "Analyse des locuteurs" in rendered

@pytest.mark.parametrize("invalid_segments", [
    [{"speaker": "A", "start": 0, "end": 1}] # Manque 'duration' (sera p-e absent si le code python change)
])
def test_diarization_view_robustness(templates, invalid_segments):
    """
    Vérifie la robustesse du template si certaines clés manquent.
    """
    mock_request = MagicMock()
    context = {
        "request": mock_request,
        "segments": invalid_segments,
        "filename": "robust.wav",
        "client_id": "user_mock_123"
    }
    template = templates.get_template("diarization_view.html")
    # Si 'duration' est absent, Jinja2 devrait renvoyer une chaîne vide ou lever UndefinedError selon config
    # Mais ici on veut surtout voir si ça plante avec "unhashable type: dict"
    rendered = template.render(context)
    assert "Diarisation" in rendered
