import re
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("ffmpeg")

def test_model_selector_has_selected_option():
    """
    Vérifie que la page d'accueil rend le sélecteur de modèle avec
    au moins une option marquée `selected`.
    """
    from backend.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200, "La requête GET / a échoué"

    html = response.text

    # Extraire le bloc <select id="modelSelector">...</select>
    select_match = re.search(
        r'<select[^>]*id="modelSelector"[^>]*>(.*?)</select>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    assert select_match, "Le sélecteur <select id=\"modelSelector\"> est introuvable dans le HTML"

    options_html = select_match.group(1)

    # Vérifier qu'au moins une option possède l'attribut `selected`
    assert re.search(r'<option[^>]*selected[^>]*>', options_html, re.IGNORECASE), (
        "Aucune option du sélecteur de modèle n'est marquée comme sélectionnée"
    )