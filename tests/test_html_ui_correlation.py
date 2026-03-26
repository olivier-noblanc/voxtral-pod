import pathlib
from check_dom_contract import extract_html_ids

def _paths():
    """Return (html_ui_path, static_path) for the test."""
    project_root = pathlib.Path(__file__).resolve().parent.parent
    html_ui_path = project_root / "backend" / "html_ui.py"
    static_path = project_root / "static" / "index.html"
    assert html_ui_path.exists(), f"Fichier introuvable : {html_ui_path}"
    assert static_path.exists(), f"Fichier introuvable : {static_path}"
    return html_ui_path, static_path

def test_html_ui_includes_app_js():
    """Vérifie que le template html_ui.py charge bien le script app.js."""
    html_ui_path, _ = _paths()
    content = html_ui_path.read_text(encoding="utf-8")
    assert '<script src="/static/app.js"' in content, "Le script app.js n’est pas inclus dans html_ui.py"

def test_html_ui_and_static_have_same_ids():
    """Vérifie que les IDs présents dans html_ui.py et static/index.html sont identiques."""
    html_ui_path, static_path = _paths()
    html_ui_code = html_ui_path.read_text(encoding="utf-8")
    static_code = static_path.read_text(encoding="utf-8")
    ids_ui = extract_html_ids(html_ui_code)
    ids_static = extract_html_ids(static_code)
    assert ids_ui == ids_static, (
        f"Les IDs diffèrent entre html_ui.py et static/index.html : "
        f"{ids_ui.symmetric_difference(ids_static)}"
    )