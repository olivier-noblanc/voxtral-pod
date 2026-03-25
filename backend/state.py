import os

# Base de données des jobs (FIFO)
jobs_db = {}

# Taille maximale de la base de données
JOBS_DB_MAX_SIZE = 500

def add_job(job_id, data):
    """
    Ajoute un job à la base de données ``jobs_db``.
    Si la taille maximale est atteinte, le job le plus ancien est supprimé
    (rotation FIFO) avant d’ajouter le nouveau.

    Args:
        job_id (str): Identifiant unique du job.
        data (any): Données associées au job.
    """
    # Supprimer le plus ancien si la capacité maximale est atteinte
    if len(jobs_db) >= JOBS_DB_MAX_SIZE:
        # ``dict`` conserve l’ordre d’insertion (Python ≥3.7)
        oldest_key = next(iter(jobs_db))
        jobs_db.pop(oldest_key, None)
    jobs_db[job_id] = data

# Nom du modèle ASR à utiliser, récupéré depuis l’environnement
model_name = os.getenv("ASR_MODEL", "whisper").lower()

# Instance du moteur ASR (initialisée au démarrage de l’application)
asr_engine = None