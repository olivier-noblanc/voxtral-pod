from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Chemin vers la base de données SQLite (par défaut dans le répertoire du projet)
DB_PATH = os.getenv("DATABASE_URL", os.path.join(os.path.dirname(__file__), "..", "jobs.db"))

# Taille maximale de la table jobs (rotation FIFO)
JOBS_DB_MAX_SIZE = 500

def get_db() -> sqlite3.Connection:
    """Retourne une connexion SQLite avec les options recommandées."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except sqlite3.Error:
        pass  # noqa: S110 - Fallback si WAL n'est pas supporté
    return conn

def cleanup_stuck_jobs() -> None:
    """
    Parcourt la base SQLite au démarrage pour trouver les jobs restés en 
    'uploading' ou 'processing:...' (interrompus par un crash/redémarrage).
    Les marque comme 'erreur' pour que le client ne reste pas bloqué indéfiniment.
    """
    try:
        with get_db() as conn:
            cursor = conn.execute("SELECT job_id, data FROM jobs")
            rows = cursor.fetchall()
            for row in rows:
                try:
                    data = json.loads(row['data'])
                    status = data.get("status", "")
                    if status not in ("terminé", "not_found", "erreur") and (
                        status.startswith("processing:") or status == "uploading"
                    ):
                        data["status"] = "erreur"
                        data["error_details"] = "Job interrompu (redémarrage du serveur)"
                        conn.execute(
                            "UPDATE jobs SET data=? WHERE job_id=?",
                            (json.dumps(data, ensure_ascii=False), row['job_id'])
                        )
                except Exception as e:
                    job_id = row.get('job_id', 'unknown')
                    logger.error(
                        f"Error processing job {job_id} during stuck cleanup: {e}"
                    )
            conn.commit()
    except Exception as e:
        print(f"Erreur lors du nettoyage des jobs SQLite : {e}")

def cleanup_stale_jobs(hours: int = 4) -> None:
    """
    Marque les jobs 'uploading' ou 'processing:...' plus vieux que 'hours'
    comme 'erreur' (jobs stale/abandonnés).
    """
    try:
        with get_db() as conn:
            cursor = conn.execute("SELECT job_id, data, created_at FROM jobs")
            rows = cursor.fetchall()
            # Use naive UTC datetime for compatibility
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
            for row in rows:
                try:
                    created_at_str = row['created_at']
                    # SQLite stores timestamps as TEXT in UTC
                    created_at = datetime.datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                    if created_at >= cutoff:
                        continue
                    data = json.loads(row['data'])
                    status = data.get("status", "")
                    if status not in ("terminé", "not_found", "erreur") and (
                        status.startswith("processing:") or status == "uploading"
                    ):
                        data["status"] = "erreur"
                        data["error_details"] = f"Job expiré (stale après {hours}h)"
                        conn.execute(
                            "UPDATE jobs SET data=? WHERE job_id=?",
                            (json.dumps(data, ensure_ascii=False), row['job_id'])
                        )
                except Exception as e:
                    logger.error(
                        f"Error processing job {row['job_id']} during stale cleanup: {e}"
                    )
            conn.commit()
    except Exception as e:
        print(f"Erreur lors du nettoyage des jobs stale : {e}")

def init_db() -> None:
    """Initialise les tables SQLite si elles n'existent pas."""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        cursor = conn.execute("SELECT value FROM config WHERE key='current_model'")
        if not cursor.fetchone():
            default_model = os.getenv("ASR_MODEL", "whisper").lower()
            conn.execute(
                "INSERT INTO config (key, value) VALUES ('current_model', ?)",
                (default_model,)
            )
        conn.commit()
    cleanup_stuck_jobs()

# init_db()  # L'initialisation est différée à l'exécution ; appelée explicitement où nécessaire.

def get_current_model() -> str:
    """
    Retourne le modèle actuel. Initialise la base de données si nécessaire.
    """
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key='current_model'"
            ).fetchone()
            return row['value'] if row else "whisper"
    except sqlite3.OperationalError:
        # La table n'existe pas encore – on initialise la DB puis on réessaye
        init_db()
        with get_db() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key='current_model'"
            ).fetchone()
            return row['value'] if row else "whisper"

def set_current_model(model: str) -> None:
    """Met à jour le modèle courant dans la configuration."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('current_model', ?)",
            (model,)
        )
        conn.commit()

def add_job(job_id: str, data: Dict[str, Any]) -> None:
    """
    Ajoute un job dans la base de données SQLite.
    (Rotation FIFO si max size atteint)
    """
    with get_db() as conn:
        cursor = conn.execute('SELECT COUNT(*) FROM jobs')
        count = cursor.fetchone()[0]
        if count >= JOBS_DB_MAX_SIZE:
            conn.execute('''
                DELETE FROM jobs WHERE job_id IN (
                    SELECT job_id FROM jobs ORDER BY created_at ASC LIMIT 1
                )
            ''')
        conn.execute(
            'INSERT OR REPLACE INTO jobs (job_id, data) VALUES (?, ?)',
            (job_id, json.dumps(data, ensure_ascii=False))
        )
        conn.commit()

_GET_JOB_SENTINEL = object()

def get_job(job_id: str, default: Any = _GET_JOB_SENTINEL) -> Dict[str, Any]:
    """
    Récupère le job identifié par ``job_id``.
    Retourne le dictionnaire JSON stocké ou ``default`` si le job n'existe pas.
    """
    with get_db() as conn:
        cursor = conn.execute('SELECT data FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        if row:
            res: dict[str, Any] = json.loads(row['data'])
            return res
    if default is _GET_JOB_SENTINEL:
        return {}
    return default  # type: ignore[no-any-return]

def update_job(job_id: str, data_update: Dict[str, Any]) -> None:
    """
    Met à jour (ou crée) le job ``job_id`` avec les champs fournis dans ``data_update``.
    """
    with get_db() as conn:
        conn.execute('BEGIN IMMEDIATE')  # lock
        cursor = conn.execute('SELECT data FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        if row:
            existing = json.loads(row['data'])
            existing.update(data_update)
            conn.execute(
                'UPDATE jobs SET data = ? WHERE job_id = ?',
                (json.dumps(existing, ensure_ascii=False), job_id)
            )
        else:
            conn.execute(
                'INSERT INTO jobs (job_id, data) VALUES (?, ?)',
                (job_id, json.dumps(data_update, ensure_ascii=False))
            )
        conn.commit()

# Global cache for the ASR engine
model_name: Optional[str] = None
asr_engine: Any = None

def get_asr_engine(load_model: bool = False) -> Any:
    """
    Retourne l'instance asr_engine pour le worker actuel.
    Si le modèle a changé dans la configuration SQLite, il recrée l'instance.
    """
    global asr_engine, model_name
    from backend.core.engine import SotaASR

    target_model = get_current_model()

    if asr_engine is None or model_name != target_model:
        model_name = target_model
        asr_engine = SotaASR(model_id=model_name, hf_token=os.getenv("HF_TOKEN"))

    if load_model and not getattr(asr_engine, "_loaded", False):
        asr_engine.load()

    return asr_engine
