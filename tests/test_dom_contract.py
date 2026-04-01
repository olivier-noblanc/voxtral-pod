import pathlib
import sys

# Racine du projet = dossier parent de tests/
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from check_dom_contract import (
    extract_html_ids,
    extract_js_ids,
    find_add_event_listener_issues,
    find_ws_send_issues,
)

# Source de vérité HTML — doit être la même que check_dom_contract.py
# Si html_ui.py génère le HTML dynamiquement, c'est lui la source, pas index.html
_HTML_CANDIDATES = [
    _PROJECT_ROOT / "backend" / "html_ui.py",
    _PROJECT_ROOT / "backend" / "templates" / "index.html",
]


def _paths() -> tuple[pathlib.Path, pathlib.Path]:
    """Retourne (js_path, html_path) — même résolution que check_dom_contract.py."""
    js_path = _PROJECT_ROOT / "static" / "app.js"
    assert js_path.exists(), f"Fichier introuvable : {js_path}"

    # Prendre la première source HTML qui existe, priorité à html_ui.py
    html_path = next((p for p in _HTML_CANDIDATES if p.exists()), None)
    assert html_path is not None, (
        f"Aucune source HTML trouvée parmi : {[str(p) for p in _HTML_CANDIDATES]}"
    )

    # Vérifier la cohérence — si les deux existent, ils doivent avoir les mêmes IDs
    existing = [p for p in _HTML_CANDIDATES if p.exists()]
    if len(existing) > 1:
        ids_per_file = {p: extract_html_ids(p.read_text(encoding="utf-8")) for p in existing}
        all_ids = list(ids_per_file.values())
        divergence = all_ids[0].symmetric_difference(all_ids[1])
        assert not divergence, (
            f"Les sources HTML sont désynchronisées — IDs divergents : {divergence}\n"
            f"  {existing[0].name} vs {existing[1].name}\n"
            f"  Supprimer l'une ou régénérer index.html depuis html_ui.py."
        )

    return js_path, html_path


def test_html_sources_are_consistent() -> None:
    """
    Si html_ui.py ET index.html coexistent, leurs IDs doivent être identiques.
    Détecte les cas où un LLM modifie l'un sans mettre à jour l'autre.
    """
    _paths()  # La vérification est dans _paths()


def test_all_js_ids_exist_in_html() -> None:
    """Vérifie que chaque ID utilisé dans le JS existe dans le HTML."""
    js_path, html_path = _paths()
    js_code = js_path.read_text(encoding="utf-8")
    html_code = html_path.read_text(encoding="utf-8")

    missing_ids = extract_js_ids(js_code) - extract_html_ids(html_code)
    assert missing_ids == set(), (
        f"IDs référencés dans app.js mais absents du HTML : {missing_ids}\n"
        f"Source HTML : {html_path}\n"
        f"Le composant HTML correspondant a probablement été supprimé sans mettre à jour le JS."
    )


def test_no_silent_fail_addeventlistener() -> None:
    """Vérifie l'absence du pattern addEventListener sans log d'erreur."""
    js_path, _ = _paths()
    js_code = js_path.read_text(encoding="utf-8")

    issues = find_add_event_listener_issues(js_code)
    assert not issues, (
        f"addEventListener anti-pattern détecté sur les variables : {issues}"
    )


def test_websocket_send_has_readystate_check() -> None:
    """Vérifie que chaque ws.send() vérifie readyState === 1."""
    js_path, _ = _paths()
    js_code = js_path.read_text(encoding="utf-8")

    issues = find_ws_send_issues(js_code)
    assert not issues, (
        f"ws.send() sans vérification readyState détecté aux lignes : {issues}"
    )
