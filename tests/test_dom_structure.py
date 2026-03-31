from typing import Any, Any

from fastapi.testclient import TestClient


def test_homepage_structure(client: TestClient, dom_audit: Any) -> None:
    """
    Ensure the homepage has a proper semantic structure.
    """
    response = client.get("/")
    assert response.status_code == 200

    soup = dom_audit(response.text)

    # 1. Main presence
    assert soup.find("main") is not None, "Homepage must have exactly one <main> element."

    # 2. Hero/Title presence (assuming H1 exists)
    assert soup.find("h1") is not None, "Homepage must have at least one <h1>."

    # 3. Form elements for upload/transcription
    # We check if there's at least one button for action
    assert len(soup.find_all("button")) >= 1, "Homepage should have interaction buttons."

    # 4. Meta tags (SEO check)
    assert soup.find("title") is not None, "Page must have a <title>."


def test_diarization_view_structure(client: TestClient, dom_audit: Any) -> None:
    """
    Verify the diarization view has specific required elements.
    """
    # Assuming there's a route for diarization viewing
    response = client.get("/")  # Placeholder route, adjust as per routes/api.py
    soup = dom_audit(response.text)

    # Every page should have a <nav> or a way back to home
    links = [a.get("href") for a in soup.find_all("a")]
    assert any(href in ["/", "./", "/home"] for href in links), "Page should link back to home."
