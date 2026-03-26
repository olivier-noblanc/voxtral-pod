import os
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "jobs.db")

JOBS_DB_MAX_SIZE = 500

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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
            conn.execute("INSERT INTO config (key, value) VALUES ('current_model', ?)", (default_model,))
        conn.commit()


# init_db()  # Removed automatic DB initialization at import time

def get_current_model():
    """
    Retourne le modèle actuel. Initialise la base de données si nécessaire.
    """
    try:
        with get_db() as conn:
            row = conn.execute("SELECT value FROM config WHERE key='current_model'").fetchone()
            return row['value'] if row else "whisper"
    except sqlite3.OperationalError:
        # La table n'existe pas encore – on initialise la DB puis on réessaye
        init_db()
        with get_db() as conn:
            row = conn.execute("SELECT value FROM config WHERE key='current_model'").fetchone()
            return row['value'] if row else "whisper"

def set_current_model(model):
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES ('current_model', ?)", (model,))
        conn.commit()

def add_job(job_id, data):
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

def get_job(job_id, default=None):
    with get_db() as conn:
        cursor = conn.execute('SELECT data FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row['data'])
    return default if default is not None else {}

def update_job(job_id, data_update):
    with get_db() as conn:
        conn.execute('BEGIN IMMEDIATE') # lock
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

model_name = None
asr_engine = None

def get_asr_engine(load_model=False):
    """
    Retourne l'instance asr_engine pour le worker actuel.
    Si le modèle a changé dans la config globale (SQLite), il recrée l'instance.
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