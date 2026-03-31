import sys
import pathlib
from typing import Any, Callable, Generator
import pytest
from fastapi.testclient import TestClient

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.main import app

@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def dom_audit() -> Callable[[str], Any]:
    """Provides a BeautifulSoup instance for HTML parsing in tests."""
    from bs4 import BeautifulSoup
    def _audit(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")
    return _audit
