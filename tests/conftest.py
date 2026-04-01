import asyncio
import pathlib
import socket
import sys
import threading
import time
from typing import Any, Callable, Generator

import pytest
import uvicorn
from fastapi.testclient import TestClient
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import os
os.environ["TESTING"] = "1"

from backend.main import app

# ---------------------------------------------------------------------------
# Fixtures HTTP (TestClient — rapide, sans navigateur)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """TestClient FastAPI — pour tous les tests HTTP unitaires."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dom_audit() -> Callable[[str], Any]:
    """BeautifulSoup pour parser le HTML retourné par le serveur."""
    from bs4 import BeautifulSoup
    def _audit(html: str) -> Any:
        return BeautifulSoup(html, "html.parser")
    return _audit


# ---------------------------------------------------------------------------
# Fixtures Playwright (navigateur headless — tests e2e UI)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def live_server_url() -> Generator[str, None, None]:
    """
    Lance l'app FastAPI dans un thread avec sa propre event loop.
    Évite le conflit asyncio entre uvicorn et pytest.
    """
    host, port = "127.0.0.1", 8001

    def run_server() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            app, host=host, port=port,
            log_level="error",
            loop="none",
            ws="none",  # désactive WebSocket pour éviter le warning websockets.legacy
        )
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Attendre que le serveur soit prêt (max 5s)
    for _ in range(50):
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    yield f"http://{host}:{port}"
    # daemon=True — le thread s'arrête avec le process pytest


@pytest.fixture(scope="session")
def browser_context(live_server_url: str):
    """Contexte Playwright partagé sur toute la session."""
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context()
        yield context, live_server_url
        context.close()
        browser.close()


@pytest.fixture
def page(browser_context) -> Generator[tuple[Page, str], None, None]:
    """
    Une page Playwright fraîche par test.
    Usage : def test_foo(page): p, base_url = page
    """
    context, base_url = browser_context
    p = context.new_page()
    yield p, base_url
    p.close()
