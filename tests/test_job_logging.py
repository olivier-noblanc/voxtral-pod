import json

from backend.state import add_job, append_job_log, get_db, get_job_log


def test_job_logging_regression() -> None:
    """
    Test de non-régression pour s'assurer que append_job_log et get_job_log
    fonctionnent correctement et n'échouent pas via un problème d'importation.
    """
    job_id = "test_log_job_123"
    
    # 1. Initialiser le job
    add_job(job_id, {"status": "processing", "progress": 10})
    
    # 2. Append quelques logs
    append_job_log(job_id, "Début du traitement")
    append_job_log(job_id, "Tâche 1 terminée")
    
    # 3. Récupérer et vérifier
    log_content = get_job_log(job_id)
    assert "Début du traitement" in log_content
    assert "Tâche 1 terminée" in log_content
    
    # Vérifie que les timestamps ont été insérés
    assert "[" in log_content
    assert "]" in log_content

    # 4. Vérification base de données brute
    conn = get_db()
    cur = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,))
    row = cur.fetchone()
    assert row is not None
    data_dict = json.loads(row[0])
    
    assert "logs" in data_dict
    assert len(data_dict["logs"]) == 2
    assert "Début du traitement" in data_dict["logs"][0]
