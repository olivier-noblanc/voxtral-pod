from typing import Any

def test_home_route_returns_html(client: Any) -> None:
    """
    Régression : la route d’accueil doit renvoyer du HTML via HTMLResponse
    et contenir le token « GPU: » ajouté par le serveur.
    """
    response = client.get("/")
    # Le statut doit être 200 OK
    assert response.status_code == 200
    # Le type de contenu doit être du HTML
    assert "text/html" in response.headers.get("content-type", "")
    # Le corps doit contenir le préfixe ajouté pour les tests
    assert "GPU:" in response.text
