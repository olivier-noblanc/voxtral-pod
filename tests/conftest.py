import os
import sqlite3
import threading
import time
import asyncio
import tempfile
import nest_asyncio
from pathlib import Path
from typing import Generator, List, Optional, Any, Dict

# 1. Force the database path for tests IMMEDIATELY
os.environ["TESTING"] = "1"
# Ensure we use a clean temp directory for the database path
import tempfile
_test_db_dir = tempfile.mkdtemp(prefix="voxtral_test_")
_test_db_path = os.path.join(_test_db_dir, "test_jobs.db")
os.environ["DATABASE_URL"] = _test_db_path

# 2. Apply nest_asyncio
import nest_asyncio
nest_asyncio.apply()

# 3. NOW import the backend components
from backend.main import app
from backend.state import init_db

# 4. Initialize the DB immediately
init_db()

import pytest
import uvicorn
from fastapi.testclient import TestClient
from bs4 import BeautifulSoup


def pytest_configure(config: Any) -> None:
    """Initial pytest configuration."""
    # S'assure que nest_asyncio est appliqué même si importé plus tard
    nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Configuration Globale & Nettoyage
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_test_env() -> None:
    """Initialize the test environment."""
    # Les variables d'env sont déjà gérées au niveau module pour la stabilité uvicorn/threads
    pass


@pytest.fixture(scope="session")
def event_loop_policy() -> Any:
    """Force use of ProactorEventLoopPolicy on Windows."""
    if os.name == 'nt':
        # On évite le selector loop car le proactor est nécessaire pour subprocesses/named pipes
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def event_loop(event_loop_policy: Any) -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Surcharge de la fixture event_loop pour assurer l'isolement total par test.
    Elle gère explicitement sa fermeture pour éviter les warnings.
    """
    asyncio.set_event_loop_policy(event_loop_policy)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Re-apply nest_asyncio on each created loop
    nest_asyncio.apply(loop)
    
    yield loop
    
    # Standard cleanup for pytest-asyncio 0.21 on Windows
    try:
        if not loop.is_closed():
            # Proper shutdown of async generators and clearing and pending tasks
            loop.run_until_complete(loop.shutdown_asyncgens())
            # Briefly run to clear any microtasks (like callbacks)
            loop.stop()
            loop.close()
    except Exception:
        # On Windows Proactor can sometimes throw if already being cleaned up by asyncio.run
        pass
    finally:
        asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Fixtures HTTP (TestClient — rapide, sans serveur réel)
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient — for all unit HTTP tests (isolated by function)."""
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
