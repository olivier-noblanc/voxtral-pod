import re
from pathlib import Path

APP_JS_PATH = Path(__file__).parents[1] / "static" / "app.js"
INDEX_HTML_PATH = Path(__file__).parents[1] / "backend" / "templates" / "index.html"

def test_ai_buttons_exist_in_frontend():
    """Vérifie que les boutons pour solliciter l'IA de post-traitement (Albert) 
    sont bien présents dans la génération de l'Historique, et que les handlers existent."""
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    
    # HTML Buttons injection
    assert 'generateSummary(this)' in js_code, "Régression: le bouton 'Compte Rendu' a disparu de l'historique."
    assert 'generateActions(this)' in js_code, "Régression: le bouton 'Actions' a disparu de l'historique."

    # JS Implementation
    assert 'async function generateSummary' in js_code, "Régression: implémentation manquante pour generateSummary."
    assert 'async function generateActions' in js_code, "Régression: implémentation manquante pour generateActions."

def test_model_selector_is_wired():
    """Vérifie que le menu de sélection de modèle ASR est fonctionnel et envoie 
    bien sa valeur au backend lorsqu'il change."""
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    html_code = INDEX_HTML_PATH.read_text(encoding="utf-8")

    # Présence dans le DOM
    assert 'id="modelSelector"' in html_code, "Régression: Le selecteur de modèle a disparu du HTML."
    
    # Ecouteur d'évènement JS
    assert re.search(r"getElementById\(['\"]modelSelector['\"]\)", js_code) is not None, "Régression: Le JS ne cible plus le modelSelector."
    assert "addEventListener('change'" in js_code, "Régression: Le JS n'écoute plus le changement du modèle."
    
    # Appel vers l'API
    assert "fetch('/change_model" in js_code, "Régression: La logique JS ne requête plus /change_model en backend."

def test_albert_partial_checkbox():
    """Vérifier que la case à cocher des transcriptions partielles (économie d'API) est présente et exploitée."""
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    html_code = INDEX_HTML_PATH.read_text(encoding="utf-8")

    assert 'id="albertPartial"' in html_code, "Régression: La case à cocher albertPartial a été supprimée du HTML"
    assert "getElementById('albertPartial')" in js_code, "Régression: La variable albertPartial a été supprimée ou ignorée dans app.js"
