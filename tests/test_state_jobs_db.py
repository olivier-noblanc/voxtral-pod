import pytest

# The jobs database utilities are defined in backend.state
from backend.state import add_job, jobs_db, JOBS_DB_MAX_SIZE


def test_jobs_db_fifo_rotation():
    """
    Insert 501 jobs into the FIFO jobs_db (max size 500).
    Verify that the length is capped at 500 and that the oldest entry
    has been removed.
    """
    # Ensure a clean state
    jobs_db.clear()

    # Insert 501 distinct jobs
    for i in range(501):
        job_id = f"job_{i}"
        add_job(job_id, {"status": "queued", "progress": 0})

    # The database should contain exactly JOBS_DB_MAX_SIZE entries
    assert len(jobs_db) == JOBS_DB_MAX_SIZE, "jobs_db size should be capped at max size"

    # The first inserted job ("job_0") should have been evicted
    assert "job_0" not in jobs_db, "oldest job should have been removed from the FIFO"


def test_add_job_overwrites_existing():
    """
    Adding a job with an existing job_id should overwrite the previous entry
    without creating a duplicate.
    """
    jobs_db.clear()

    job_id = "duplicate_job"
    add_job(job_id, {"status": "first", "progress": 10})
    # Overwrite with new data
    add_job(job_id, {"status": "second", "progress": 20})

    # There should be exactly one entry for the job_id
    assert len(jobs_db) == 1, "jobs_db should contain a single entry for the duplicated job_id"
    # The stored data should reflect the latest addition
    stored = jobs_db[job_id]
    assert stored["status"] == "second"
    assert stored["progress"] == 20