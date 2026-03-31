from typing import Any
from fastapi.testclient import TestClient
from pytest_snapshot.plugin import Snapshot


def test_homepage_snapshot(client: TestClient, snapshot: Snapshot) -> None:
    """
    Test that the homepage renders exactly as expected.
    Use 'pytest --snapshot-update' to update snapshots.
    """
    response = client.get("/api/")
    assert response.status_code == 200

    # We strip whitespace to avoid fragile failures due to indentation changes
    html_content = response.text.strip()

    # Snapshot will be stored in tests/snapshots/test_ui_snapshot/test_homepage_snapshot.txt
    snapshot.assert_match(html_content, "homepage.html")


def test_view_page_snapshot(client: TestClient, snapshot: Snapshot) -> None:
    """
    Test that the transcription view page renders correctly.
    """
    # Note: We might need a real or mock transcription ID here
    # For now, we test the 404 or a generic view if possible
    response = client.get("/view/non_existent_id")
    # Even a 404 page should have a consistent look
    snapshot.assert_match(response.text.strip(), "view_404.html")
