from typing import Any

def test_index_includes_assets(client: Any) -> None:
    """
    Régression : le template index.html doit charger les assets essentiels
    (app.js, dsfr.min.css, custom.css).
    """
    response = client.get("/")
    assert response.status_code == 200

    html = response.text

    # Vérifier la présence du script JavaScript principal
    assert '<script src="/static/app.js"' in html

    # Vérifier la présence des feuilles de style CSS
    assert '<link rel="stylesheet" href="/static/dsfr.min.css"' in html
    assert '<link rel="stylesheet" href="/static/custom.css"' in html
