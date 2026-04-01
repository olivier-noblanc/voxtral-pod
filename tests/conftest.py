import os
import sqlite3
import threading
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Generator, List, Optional, Any, Dict

import pytest
import uvicorn
import nest_asyncio
from fastapi.testclient import TestClient
from bs4 import BeautifulSoup

# Applique nest_asyncio le plus tôt possible
nest_asyncio.apply()

from backend.main import app
from backend.state import init_db


def pytest_configure(config: Any) -> None:
    """Configuration initiale de pytest."""
    # S'assure que nest_asyncio est appliqué même si importé plus tard
    nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Configuration Globale & Nettoyage
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_test_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Initialise l'environnement de test (DB isolée, variables d'env)."""
    # Chemin vers une DB de test isolée
    test_db_dir = tmp_path_factory.mktemp("test_db")
    test_db_path = test_db_dir / "test_jobs.db"
    
    os.environ["TESTING"] = "1"
    os.environ["DATABASE_URL"] = str(test_db_path)
    
    # S'assure que la DB de test est initialisée
    init_db()


@pytest.fixture(scope="session")
def event_loop_policy() -> Any:
    """Force l'utilisation du ProactorEventLoopPolicy sur Windows."""
    if os.name == 'nt':
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def event_loop(event_loop_policy: Any) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Surcharge de la fixture event_loop pour assurer l'isolement total par test.
    """
    asyncio.set_event_loop_policy(event_loop_policy)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Ré-applique nest_asyncio sur chaque loop créé
    nest_asyncio.apply(loop)
    
    yield loop
    # Nettoyage
    loop.close()
    asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Fixtures HTTP (TestClient — rapide, sans serveur réel)
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """TestClient FastAPI — pour tous les tests HTTP unitaires (isolé par fonction)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dom_audit() -> Any:
    """Fixture pour l'audit du DOM via BeautifulSoup."""
    def _audit(html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")
    return _audit


# ---------------------------------------------------------------------------
# Fixtures Live Server (Serveur réel dans un thread séparé)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def live_server_url() -> Generator[str, None, None]:
    """
    Lance un serveur Uvicorn réel dans un thread séparé.
    Utilisé pour les tests Playwright ou WebSocket.
    """
    host = "127.0.0.1"
    port = 8001
    
    # Configuration du serveur pour permettre un shutdown propre
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    # Attend que le serveur soit prêt
    time.sleep(1)
    
    url = f"http://{host}:{port}"
    yield url
    
    # Shutdown propre du serveur Uvicorn
    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Fixtures Integration Playwright (Native via pytest-playwright)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(live_server_url: str) -> str:
    """
    Hook pour injecter live_server_url dans le plugin pytest-playwright.
    """
    return live_server_url
