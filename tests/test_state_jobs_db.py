import pytest

import backend.state as state
from backend.state import JOBS_DB_MAX_SIZE, add_job, get_job


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "DB_PATH", str(tmp_path / "test.db"))
    state.init_db()
    yield


def test_jobs_db_fifo_rotation() -> None:
    for i in range(501):
        add_job(f"job_{i}", {"status": "queued"})
    with state.get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert count == JOBS_DB_MAX_SIZE
    assert get_job("job_0") == {}


def test_add_job_overwrites_existing() -> None:
    add_job("dup", {"status": "first", "progress": 10})
    add_job("dup", {"status": "second", "progress": 20})
    with state.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE job_id = 'dup'"
        ).fetchall()
    assert len(rows) == 1
    assert get_job("dup")["status"] == "second"


def test_get_job_missing_returns_empty() -> None:
    result = get_job("nonexistent")
    assert result == {}


def test_cleanup_stuck_jobs_on_init() -> None:
    """
    Test that jobs stuck in 'uploading' or 'processing:...' states
    are properly moved to 'erreur' when the DB is re-initialized (or explicitly cleaned).
    """
    state.add_job("stuck1", {"status": "uploading", "progress": 50})
    state.add_job("stuck2", {"status": "processing:Transcription", "progress": 10})
    state.add_job("ok_job", {"status": "terminé", "progress": 100})
    state.add_job("not_found_job", {"status": "not_found", "progress": 0})
    state.add_job("error_job", {"status": "erreur", "progress": 0})

    state.cleanup_stuck_jobs()

    # Verify stuck jobs are now errors
    stuck1 = get_job("stuck1")
    assert stuck1["status"] == "erreur"
    assert "Job interrompu" in stuck1.get("error_details", "")

    stuck2 = get_job("stuck2")
    assert stuck2["status"] == "erreur"

    # Verify other jobs are untouched
    assert get_job("ok_job")["status"] == "terminé"
    assert get_job("not_found_job")["status"] == "not_found"
    assert get_job("error_job")["status"] == "erreur"
