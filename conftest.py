"""
Configuration for pytest.
"""
from backend.state import init_db

def pytest_sessionstart(_session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    Ensures the database is initialized before any tests run.
    """
    init_db()
