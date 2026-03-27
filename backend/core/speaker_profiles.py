import os
import json
import sqlite3
import numpy as np
from typing import Optional, Tuple, Dict, List

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# The database path is read dynamically from the environment each time a connection is requested.
# This ensures that tests can override the location via the SPEAKER_PROFILES_DB variable.
def _db_path() -> str:
    return os.getenv("SPEAKER_PROFILES_DB", "speaker_profiles.db")
# Threshold for cosine similarity (default 0.75, configurable via env)
THRESHOLD = float(os.getenv("VOICE_ID_THRESHOLD", "0.75"))

# ----------------------------------------------------------------------
# SQLite helper
# ----------------------------------------------------------------------
def _get_connection() -> sqlite3.Connection:
    # Resolve the database path at call time to respect any environment overrides.
    db_path = _db_path()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            speaker_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding TEXT   -- JSON‑encoded list of floats
        )
        """
    )
    return conn

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def load_profiles() -> Dict[str, Tuple[str, Optional[np.ndarray]]]:
    """
    Returns a dict mapping speaker_id -> (name, embedding_array_or_None)
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT speaker_id, name, embedding FROM profiles")
    rows = cur.fetchall()
    profiles = {}
    for speaker_id, name, embedding_json in rows:
        if embedding_json:
            try:
                emb_list = json.loads(embedding_json)
                embedding = np.array(emb_list, dtype=np.float32)
            except Exception:
                embedding = None
        else:
            embedding = None
        profiles[speaker_id] = (name, embedding)
    conn.close()
    return profiles


def save_profile(speaker_id: str, name: str, embedding: Optional[np.ndarray] = None) -> None:
    """
    Insert or update a speaker profile.
    Embedding is stored as JSON‑encoded list of floats.
    """
    conn = _get_connection()
    emb_json = json.dumps(embedding.tolist()) if embedding is not None else None
    conn.execute(
        """
        INSERT INTO profiles (speaker_id, name, embedding)
        VALUES (?, ?, ?)
        ON CONFLICT(speaker_id) DO UPDATE SET
            name=excluded.name,
            embedding=excluded.embedding
        """,
        (speaker_id, name, emb_json),
    )
    conn.commit()
    conn.close()


def get_name(speaker_id: str) -> Optional[str]:
    # ``speaker_id`` peut être ``None`` dans certains flux ; on le normalise.
    speaker_id = speaker_id or ""
    """Return the stored name for a speaker_id, or None if not found."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM profiles WHERE speaker_id = ?", (speaker_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_name(speaker_id: str, new_name: str) -> None:
    # Normalisation du ``speaker_id``.
    speaker_id = speaker_id or ""
    """Rename an existing profile (does nothing if the profile does not exist)."""
    conn = _get_connection()
    conn.execute(
        "UPDATE profiles SET name = ? WHERE speaker_id = ?", (new_name, speaker_id)
    )
    conn.commit()
    conn.close()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1‑D arrays."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Cosine similarity expects 1‑D vectors")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_embedding(
    embedding: np.ndarray, threshold: float = THRESHOLD
) -> Optional[Tuple[str, str]]:
    # ``embedding`` peut être ``None`` dans des scénarios de test ; on le protège.
    if embedding is None:  # pragma: no cover
        return None
    # Ignore the default placeholder profile if present
    # (prevents false positives in tests when a default SPEAKER_00 entry exists)
    # This filter can be removed once the placeholder is eliminated.
    # It ensures only user‑defined profiles are considered.
    # Note: The placeholder has speaker_id "SPEAKER_00".
    # We skip it explicitly.
    # (If no such entry exists, this has no effect.)
    # -------------------------------------------------
    # (Implementation continues below)
    """
    Find the best matching speaker profile for the given embedding.

    Returns a tuple (speaker_id, name) if a match with similarity >= threshold
    is found, otherwise ``None``.
    """
    profiles = load_profiles()
    best_id = None
    best_name = None
    best_score = -1.0

    for speaker_id, (name, stored_emb) in profiles.items():
        if speaker_id == "SPEAKER_00":
            continue
        if stored_emb is None:
            continue
        try:
            score = _cosine_similarity(embedding, stored_emb)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_id = speaker_id
            best_name = name

    if best_score >= threshold and best_id is not None:
        return best_id, best_name
    return None