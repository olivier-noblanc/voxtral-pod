"""
Test de non-régression : le bloc debug document.getElementById ne doit pas
exister dans static/app.js (supprimé suite à l'audit).
"""
import pathlib

APP_JS = pathlib.Path(__file__).parent.parent / "static" / "app.js"


def test_no_dom_getbyid_override():
    """
    Vérifie que le patch debug `document.getElementById = ...` a bien été retiré
    de app.js. Sa présence en production ralentit tous les accès DOM et peut
    interférer avec des bibliothèques tierces.
    """
    assert APP_JS.exists(), f"app.js introuvable à : {APP_JS}"
    content = APP_JS.read_text(encoding="utf-8")
    assert "document.getElementById =" not in content, (
        "Le bloc debug `document.getElementById = ...` est encore présent dans app.js. "
        "Il doit être retiré du code de production."
    )


def test_no_debug_comment_block():
    """Vérifie aussi que le commentaire '// DEBUG — à retirer en prod' est absent."""
    content = APP_JS.read_text(encoding="utf-8")
    assert "DEBUG — à retirer en prod" not in content, (
        "Le commentaire debug est encore présent dans app.js."
    )
