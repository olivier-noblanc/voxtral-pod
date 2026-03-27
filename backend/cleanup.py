import os
import time
import shutil
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor

from backend.config import TRANSCRIPTIONS_DIR, TEMP_DIR, CLEANUP_RETENTION_DAYS
from backend.state import get_db

def clean_old_jobs(days: int) -> int:
    """Removes SQLite jobs older than 'days'."""
    deleted_count = 0
    try:
        # SQLite datetime('now', '-90 days') will work reliably
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE created_at < datetime('now', ?)",
                (f"-{days} days",)
            )
            deleted_count = cursor.rowcount
            conn.commit()
    except Exception as e:
        print(f"Erreur lors du nettoyage des vieux jobs SQLite : {e}")
    return deleted_count

def clean_old_files(days: int) -> int:
    """
    Remove files in TRANSCRIPTIONS_DIR and TEMP_DIR older than 'days'.
    Recursively scans and deletes .wav, .txt, .md, .json.
    Also removes entire empty user directories or old batch chunks in TEMP_DIR.
    """
    deleted_count = 0
    now = time.time()
    cutoff = now - (days * 86400)

    directories_to_check = [
        TRANSCRIPTIONS_DIR,
        os.path.join(TRANSCRIPTIONS_DIR, "live_audio"),
        os.path.join(TRANSCRIPTIONS_DIR, "batch_audio"),
        TEMP_DIR
    ]

    for base_dir in directories_to_check:
        if not os.path.exists(base_dir):
            continue
        
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    # In TEMP_DIR we clean everything old (chunks, audio_full.wav)
                    # In TRANSCRIPTIONS_DIR we clean our specific extensions
                    allowed_exts = (".wav", ".txt", ".md", ".json")
                    is_temp = root.startswith(os.path.abspath(TEMP_DIR)) or root.startswith(TEMP_DIR)
                    is_target_ext = filename.lower().endswith(allowed_exts)

                    if is_temp or is_target_ext:
                        mtime = os.path.getmtime(filepath)
                        if mtime < cutoff:
                            os.remove(filepath)
                            deleted_count += 1
                except Exception as e:
                    print(f"Erreur lors de la suppression de {filepath}: {e}")

            # Nettoyer les dossiers vides (sauf les dossiers de base)
            for dirname in dirs:
                dirpath = os.path.join(root, dirname)
                try:
                    if not os.listdir(dirpath):
                        os.rmdir(dirpath)
                except Exception:
                    pass

    return deleted_count

def _compress_single_file(filepath: str) -> bool:
    """Helper to compress a single wav file to mp3. Returns True if successful."""
    try:
        mp3_path = os.path.splitext(filepath)[0] + ".mp3"
        if not os.path.exists(mp3_path):
            # Using -threads 0 to allow ffmpeg to use all available CPU cores for a single file compression
            result = subprocess.run(
                ["ffmpeg", "-y", "-threads", "0", "-i", filepath, "-b:a", "64k", mp3_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if result.returncode == 0 and os.path.exists(mp3_path):
                os.remove(filepath)
                return True
            else:
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
    except Exception as e:
        print(f"Erreur lors de la compression de {filepath}: {e}")
    return False

def compress_old_wavs(days: float = 0.1) -> int:
    """
    Compress .wav files older than 'days' (default 0.1 day = 2.4 hours)
    to .mp3 in parallel using multi-CPU threads.
    """
    compressed_count = 0
    now = time.time()
    cutoff = now - (days * 86400)

    directories_to_check = [
        os.path.join(TRANSCRIPTIONS_DIR, "live_audio"),
        os.path.join(TRANSCRIPTIONS_DIR, "batch_audio")
    ]

    files_to_compress = []
    for base_dir in directories_to_check:
        if not os.path.exists(base_dir):
            continue
        
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if filename.lower().endswith(".wav"):
                    filepath = os.path.join(root, filename)
                    try:
                        if os.path.getmtime(filepath) < cutoff:
                            files_to_compress.append(filepath)
                    except Exception:
                        pass

    if not files_to_compress:
        return 0

    # Use ThreadPoolExecutor for parallel compression. 
    # We use 2 workers combined with ffmpeg's internal multi-threading (-threads 0)
    # for a good balance between processing multiple files and per-file speed.
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compress_single_file, files_to_compress))
        compressed_count = sum(1 for r in results if r)

    return compressed_count

def run_cleanup(days: int = CLEANUP_RETENTION_DAYS) -> dict:
    """Exécute les nettoyages base de données et fichiers, retourne les statistiques."""
    print(f"[*] Démarrage du nettoyage automatique (Rétention: {days} jours)...")
    
    # Compress old wav to mp3 first
    compressed_files = compress_old_wavs(0.1) # compress files older than ~2.4h
    
    jobs_deleted = clean_old_jobs(days)
    files_deleted = clean_old_files(days)
    
    print(f"[*] Nettoyage terminé: {jobs_deleted} jobs supprimés, {files_deleted} fichiers supprimés, {compressed_files} fichiers WAV compressés en MP3.")
    return {"jobs_deleted": jobs_deleted, "files_deleted": files_deleted, "compressed_files": compressed_files}

async def periodic_cleanup_task(days: int):
    """Tâche asynchrone tournant en boucle toutes les 24h."""
    while True:
        try:
            # Attend d'abord légèrement pour ne pas bloquer le démarrage
            await asyncio.sleep(60) 
            # Exécute le nettoyage (dans un thread pool pour ne pas bloquer l'event loop)
            await asyncio.to_thread(run_cleanup, days)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Erreur dans periodic_cleanup_task: {e}")
        
        # Pause de 24 heures (86400 secondes)
        await asyncio.sleep(86400)
