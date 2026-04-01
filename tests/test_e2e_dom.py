"""
Tests de contrat DOM — Playwright.
Vérifie que les éléments référencés dans app.js existent et fonctionnent dans le HTML rendu.
"""
import re
import pathlib
import pytest
from playwright.sync_api import Page, expect

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
JS_PATH = PROJECT_ROOT / "static" / "app.js"


def _get_js_ids() -> set[str]:
    js_code = JS_PATH.read_text(encoding="utf-8")
    ids: set[str] = set()
    ids.update(re.findall(r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""", js_code))
    ids.update(re.findall(r"""querySelector\(\s*['"]#([^'"]+)['"]\s*\)""", js_code))
    return ids


def test_all_js_ids_exist_in_rendered_html(page: Page) -> None:
    """Chaque ID référencé dans app.js doit exister dans le DOM rendu."""
    page.goto("/")

    missing = [id_ for id_ in _get_js_ids() if page.locator(f"#{id_}").count() == 0]

    assert not missing, (
        f"IDs dans app.js absents du DOM :\n"
        + "\n".join(f"  - #{id_}" for id_ in sorted(missing))
        + "\nUn composant HTML a été supprimé sans mettre à jour app.js."
    )


def test_albertoptions_visible_when_albert_selected(page: Page) -> None:
    """
    Quand l'utilisateur sélectionne Albert dans le dropdown,
    le panneau albertOptions doit devenir visible.
    Si ce test échoue : le JS changeModel() ne gère pas l'affichage de albertOptions.
    """
    page.goto("/")

    selector = page.locator("#modelSelector")
    if selector.count() == 0:
        pytest.skip("modelSelector absent du DOM")

    # Sélectionner Albert
    selector.select_option("albert")

    options_panel = page.locator("#albertOptions")
    assert options_panel.count() == 1, "#albertOptions absent du DOM"
    expect(options_panel).to_be_visible(timeout=2000)


def test_albertpartial_unchecked_when_albert_selected(page: Page) -> None:
    """
    Quand Albert est sélectionné et albertOptions visible,
    la checkbox albertPartial doit être décochée par défaut.
    Cochée = quota Albert consommé à chaque chunk VAD.
    """
    page.goto("/")

    selector = page.locator("#modelSelector")
    if selector.count() == 0:
        pytest.skip("modelSelector absent du DOM")

    selector.select_option("albert")

    checkbox = page.locator("#albertPartial")
    assert checkbox.count() == 1, "#albertPartial absent du DOM"
    assert not checkbox.is_checked(), (
        "#albertPartial est cochée par défaut — "
        "risque de consommation quota Albert à chaque session."
    )


def test_websocket_partial_false_by_default(page: Page) -> None:
    """
    Le fallback JS partial_albert doit être false.
    Vérifie directement le source app.js — pas besoin de déclencher le WS.
    """
    js_code = JS_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"partialCheckbox\s*\?\s*partialCheckbox\.checked\s*:\s*(true|false)",
        js_code
    )
    assert match, (
        "Pattern fallback partial_albert introuvable dans app.js.\n"
        "Attendu : partialCheckbox ? partialCheckbox.checked : false"
    )
    assert match.group(1) == "false", (
        f"Fallback partial_albert est '{match.group(1)}' — doit être 'false'.\n"
        f"Chaque session live consomme du quota Albert inutilement."
    )
