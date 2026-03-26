import pytest
import backend.state as state
from backend.state import add_job, get_job, JOBS_DB_MAX_SIZE


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "test.db"))
    state.init_db()
    yield


def test_jobs_db_fifo_rotation():
    for i in range(501):
        add_job(f"job_{i}", {"status": "queued"})
    with state.get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == JOBS_DB_MAX_SIZE
    assert get_job("job_0") == {}


def test_add_job_overwrites_existing():
    add_job("dup", {"status": "first", "progress": 10})
    add_job("dup", {"status": "second", "progress": 20})
    with state.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE job_id = 'dup'"
        ).fetchall()
    assert len(rows) == 1
    assert get_job("dup")["status"] == "second"


def test_get_job_missing_returns_empty():
    result = get_job("nonexistent")
    assert result == {}