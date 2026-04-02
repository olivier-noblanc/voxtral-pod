# backend/state.py
import sqlite3
import threading
import time
import json
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
# Maximum number of job entries retained in the database (FIFO rotation)
JOBS_DB_MAX_SIZE: int = 500

# Path to the SQLite database file (can be overridden in tests)
DB_PATH = "jobs.db"

# SQLite connection with a timeout and thread safety
_connection_lock = threading.Lock()
_connection: sqlite3.Connection | None = None

def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        with _connection_lock:
            if _connection is None:
                _connection = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
                _connection.execute("PRAGMA journal_mode=WAL;")
                _connection.execute("PRAGMA foreign_keys=ON;")
                # Return rows as dict‑like objects for easier column access
                _connection.row_factory = sqlite3.Row
    return _connection

def init_db() -> None:
    conn = _get_connection()
    conn.executescript(
        '''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        '''
    )
    conn.commit()

# ----------------------------------------------------------------------
# Public accessor for the DB connection (used by cleanup and other modules)
# ----------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    """
    Expose the SQLite connection for modules that need direct access.
    """
    return _get_connection()

# ----------------------------------------------------------------------
# Cleanup helper for stale jobs (placeholder implementation)
# ----------------------------------------------------------------------
def cleanup_stale_jobs(hours: int = 4) -> None:
    """
    Remove jobs that have been in a non‑final state for longer than ``hours``.
    This is a lightweight placeholder; the actual stale‑job logic can be
    expanded later. For now it simply deletes jobs older than the given
    threshold based on the ``created_at`` timestamp.
    """
    try:
        cutoff = f"-{hours} hours"
        with get_db() as conn:
            conn.execute(
                "DELETE FROM jobs WHERE created_at < datetime('now', ?)",
                (cutoff,)
            )
            conn.commit()
    except Exception as e:
        print(f"[cleanup_stale_jobs] Erreur lors du nettoyage des jobs stale : {e}")

def add_job(job_id: str, data: str | dict) -> None:
    """
    Insert or replace a job entry.
    If ``data`` is a dict it will be stored as JSON.
    Enforces FIFO rotation based on ``JOBS_DB_MAX_SIZE``.
    """
    conn = _get_connection()
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = data
    conn.execute(
        "INSERT OR REPLACE INTO jobs (job_id, data) VALUES (?, ?)",
        (job_id, data_str),
    )
    # Enforce max size
    cur = conn.execute("SELECT COUNT(*) FROM jobs")
    count = cur.fetchone()[0]
    if count > JOBS_DB_MAX_SIZE:
        # Delete oldest entries exceeding the limit
        excess = count - JOBS_DB_MAX_SIZE
        conn.execute(
            """
            DELETE FROM jobs
            WHERE job_id IN (
                SELECT job_id FROM jobs
                ORDER BY created_at ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
    conn.commit()

def get_job(job_id: str) -> dict:
    """
    Retrieve a job. Returns an empty dict if the job does not exist.
    """
    conn = _get_connection()
    cur = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    if not row:
        return {}
    data = row[0]
    try:
        data_parsed = json.loads(data)
    except Exception:
        data_parsed = data
    return data_parsed

def delete_job(job_id: str) -> None:
    conn = _get_connection()
    conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
    conn.commit()

def list_jobs(limit: int = 500) -> List[Tuple[str, Any]]:
    """
    Return a list of jobs ordered by creation time (newest first).
    JSON data is deserialized for convenience.
    """
    conn = _get_connection()
    cur = conn.execute(
        "SELECT job_id, data FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    result: List[Tuple[str, Any]] = []
    for job_id_val, data in rows:
        try:
            data_parsed = json.loads(data)
        except Exception:
            data_parsed = data
        result.append((job_id_val, data_parsed))
    return result

def cleanup_stuck_jobs() -> None:
    """
    Convert jobs stuck in 'uploading' or 'processing:*' states to 'erreur'.
    Adds a generic error detail for 'uploading' jobs.
    """
    conn = _get_connection()
    try:
        # Update 'uploading' jobs
        conn.execute(
            """
            UPDATE jobs
            SET data = json_set(
                COALESCE(data, '{}'),
                '$.status', 'erreur',
                '$.error_details', 'Job interrompu'
            )
            WHERE json_extract(data, '$.status') = 'uploading'
            """
        )
        # Update 'processing:*' jobs
        conn.execute(
            """
            UPDATE jobs
            SET data = json_set(
                COALESCE(data, '{}'),
                '$.status', 'erreur'
            )
            WHERE json_extract(data, '$.status') LIKE 'processing:%'
            """
        )
        conn.commit()
    except Exception as e:
        print(f"[cleanup_stuck_jobs] Erreur lors du nettoyage des jobs stuck : {e}")

def get_config(key: str) -> str | None:
    conn = _get_connection()
    cur = conn.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = cur.fetchone()
    return row[0] if row else None

def set_config(key: str, value: str) -> None:
    conn = _get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()

# ----------------------------------------------------------------------
# Model selection helpers
# ----------------------------------------------------------------------
def get_current_model() -> str | None:
    """
    Retrieve the currently selected ASR model identifier from the config table.
    Returns ``None`` if not set.
    """
    return get_config("current_model")

def set_current_model(model_id: str) -> None:
    """
    Store the selected ASR model identifier in the config table.
    """
    set_config("current_model", model_id)

# Initialize the database on import
init_db()

# ----------------------------------------------------------------------
# Helper to update an existing job (used by utils and routes)
# ----------------------------------------------------------------------
def update_job(job_id: str, data: dict | str) -> None:
    """
    Update an existing job entry, merging the provided data with the stored JSON.
    If the job does not exist, it will be created.
    """
    conn = _get_connection()
    cur = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    if row:
        existing_data = row[0]
        try:
            existing_dict = json.loads(existing_data)
        except Exception:
            existing_dict = {}
        if isinstance(data, dict):
            existing_dict.update(data)
            new_data = json.dumps(existing_dict)
        else:
            new_data = data
        conn.execute("UPDATE jobs SET data = ? WHERE job_id = ?", (new_data, job_id))
    else:
        # Insert as new job if it does not exist
        if isinstance(data, dict):
            new_data = json.dumps(data)
        else:
            new_data = data
        conn.execute("INSERT INTO jobs (job_id, data) VALUES (?, ?)", (job_id, new_data))
    conn.commit()

# ----------------------------------------------------------------------
# Cached ASR engine singleton
# ----------------------------------------------------------------------
_asr_engine: Any | None = None

def get_asr_engine(load_model: bool = False) -> Any:
    """
    Return a singleton SotaASR engine.
    If ``load_model`` is True, the engine will be fully loaded.
    """
    global _asr_engine
    if _asr_engine is None:
        from backend.core.engine import SotaASR
        _asr_engine = SotaASR()
        if load_model:
            _asr_engine.load()
    return _asr_engine