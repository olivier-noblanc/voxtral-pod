import os
import sys
import pathlib
import pytest
from fastapi.testclient import TestClient

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.main import app

@pytest.fixture(scope="session")
def client():
    """Provides a TestClient instance for the FastAPI app."""
    with TestClient(app) as client:
        yield client