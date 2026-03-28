import sys
import pathlib

# Racine du projet = dossier parent de tests/
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Ajout de la racine ET de scripts/ au path
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from check_dom_contract import (
    extract_js_ids,
    extract_html_ids,
    find_add_event_listener_issues,
    find_ws_send_issues,
)


def _paths():
    """Retourne (js_path, html_path) — résolution à l'appel, pas à l'import."""
    js_path   = _PROJECT_ROOT / "static"  / "app.js"
    html_path = _PROJECT_ROOT / "backend" / "templates" / "index.html"
    assert js_path.exists(),   f"Fichier introuvable : {js_path}"
    assert html_path.exists(), f"Fichier introuvable : {html_path}"
    return js_path, html_path


def test_all_js_ids_exist_in_html():
    """Vérifie que chaque ID utilisé dans le JS existe dans le HTML."""
    js_path, html_path = _paths()
    js_code   = js_path.read_text(encoding="utf-8")
    html_code = html_path.read_text(encoding="utf-8")

    missing_ids = extract_js_ids(js_code) - extract_html_ids(html_code)
    assert missing_ids == set(), (
        f"IDs manquants dans html_ui.py : {missing_ids}"
    )


def test_no_silent_fail_addeventlistener():
    """Vérifie l'absence du pattern addEventListener sans log d'erreur."""
    js_path, _ = _paths()
    js_code = js_path.read_text(encoding="utf-8")

    issues = find_add_event_listener_issues(js_code)
    assert not issues, (
        f"addEventListener anti-pattern détecté sur les variables : {issues}"
    )


def test_websocket_send_has_readystate_check():
    """Vérifie que chaque ws.send() vérifie readyState === 1."""
    js_path, _ = _paths()
    js_code = js_path.read_text(encoding="utf-8")

    issues = find_ws_send_issues(js_code)
    assert not issues, (
        f"ws.send() sans vérification readyState détecté : {issues}"
    )