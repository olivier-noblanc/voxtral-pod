from fastapi.testclient import TestClient
from pytest_snapshot.plugin import Snapshot


def test_homepage_snapshot(client: TestClient, snapshot: Snapshot) -> None:
    """
    Test that the homepage renders exactly as expected.
    Use 'pytest --snapshot-update' to update snapshots.
    """
    response = client.get("/")
    assert response.status_code == 200

    html_content = response.text.strip()
    snapshot.assert_match(html_content, "homepage.html")


def test_view_page_snapshot(client: TestClient, snapshot: Snapshot) -> None:
    """
    Test that the transcription view page renders correctly for an unknown file.
    """
    response = client.get("/view/non_existent_id/non_existent_file.txt")
    # 404 attendu — on vérifie juste la cohérence du rendu
    snapshot.assert_match(response.text.strip(), "view_404.html")