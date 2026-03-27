import json
import pytest
import datetime
from backend.state import get_db, add_job, get_job, cleanup_stale_jobs

def test_cleanup_stale_jobs_marks_as_error():
    """
    Vérifie que cleanup_stale_jobs marque bien les vieux jobs 'uploading' comme erreur.
    """
    job_id = "test_stale_123"
    # Créer un job 'uploading'
    add_job(job_id, {"status": "uploading", "progress": 0})
    
    # Simuler une date ancienne (4h30 ago) en UTC
    with get_db() as conn:
        old_time = (datetime.datetime.utcnow() - datetime.timedelta(hours=4.5)).strftime('%Y-%m-%d %H:%M:%S')
        conn.execute("UPDATE jobs SET created_at = ? WHERE job_id = ?", (old_time, job_id))
        conn.commit()
    
    # Lancer le nettoyage
    cleanup_stale_jobs(hours=4)
    
    # Vérifier le statut
    job = get_job(job_id)
    assert job["status"] == "erreur"
    assert "expiré" in job["error_details"]

def test_cleanup_stale_jobs_ignores_recent_jobs():
    """
    Vérifie que cleanup_stale_jobs ne touche pas aux jobs récents (ex: 30 min).
    """
    job_id = "test_recent_123"
    add_job(job_id, {"status": "uploading", "progress": 10})
    
    # Lancer le nettoyage avec seuil de 4h
    cleanup_stale_jobs(hours=4)
    
    # Vérifier que le statut n'a pas changé
    job = get_job(job_id)
    assert job["status"] == "uploading"

def test_cleanup_stale_jobs_ignores_finished_jobs():
    """
    Vérifie que cleanup_stale_jobs ne touche pas aux jobs déjà terminés, 
    même s'ils sont vieux.
    """
    job_id = "test_old_finished"
    add_job(job_id, {"status": "terminé", "progress": 100})
    
    with get_db() as conn:
        old_time = (datetime.datetime.now() - datetime.timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S')
        conn.execute("UPDATE jobs SET created_at = ? WHERE job_id = ?", (old_time, job_id))
        conn.commit()
    
    cleanup_stale_jobs(hours=4)
    
    job = get_job(job_id)
    assert job["status"] == "terminé"
