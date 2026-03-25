import os
import datetime
import asyncio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from html import escape as html_escape
# import sys  # Removed unused import
from backend.html_ui import HTML_UI
from backend.core.live import LiveSession
from dulwich.repo import Repo
from dulwich import porcelain

# Import shared state (no import of server_asr_ref)
from backend.state import jobs_db, add_job, asr_engine, model_name, JOBS_DB_MAX_SIZE
from backend.config import TRANSCRIPTIONS_DIR, TEMP_DIR

router = APIRouter()

# ---------- Helper ----------
def _update_job_status(job_id: str, status: str, progress: int = 0, result_file: str | None = None):
    """Update a job entry in the FIFO jobs_db."""
    job = jobs_db.get(job_id, {})
    job.update({"status": status, "progress": progress})
    if result_file:
        job["result_file"] = result_file
    jobs_db[job_id] = job

# ---------- GET routes ----------
@router.get("/")
async def home(request: Request):
    """Render the main HTML UI."""
    try:
        repo = Repo(".")
        commit = repo.head().decode()
    except Exception:
        commit = "unknown"
    protocol = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host")
    ws_url = f"{protocol}://{host}/live?client_id={{getClientId()}}&partial_albert={{no_gpu}}"
    return HTMLResponse(
        content=HTML_UI
        .replace("{{model_name}}", model_name)
        .replace("{{device}}", "cuda" if not asr_engine.no_gpu else "cpu")
        .replace("{{commit}}", commit)
        .replace("{{ws_url}}", ws_url)
    )

@router.get("/transcriptions")
async def list_trans(client_id: str):
    """List transcription files for a client."""
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.isdir(p):
        return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    """Return a transcription file."""
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type="text/plain", filename=filename)

@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
    """Download a WAV audio file."""
    for subdir in ("live_audio", "batch_audio"):
        file_path = os.path.join(TRANSCRIPTIONS_DIR, subdir, filename)
        if os.path.isfile(file_path):
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")

@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
    """Download a transcript text file."""
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fichier transcript introuvable.")
    with open(file_path, "rb") as f:
        content = f.read()
    return Response(
        content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@router.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, request: Request):
    """Render a transcription in HTML."""
    client_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if os.path.isfile(client_path):
        file_path = client_path
        is_temp = False
    else:
        temp_path = os.path.join(TRANSCRIPTIONS_DIR, "live_audio", filename)
        if os.path.isfile(temp_path):
            file_path = temp_path
            is_temp = True
        else:
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    safe_file = html_escape(filename)
    temp_notice = "<p><em>Version temporaire (en cours de nettoyage)</em></p>" if is_temp else ""
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><title>{safe_file} – Voxtral Pod</title>
<link rel="stylesheet" href="/static/dsfr.min.css">
<style>
body {{ background: var(--background-alt-grey); }}
.transcript-container {{ max-width: 900px; margin: 2rem auto; padding: 1rem; }}
.segment {{ margin-bottom: 1.5rem; padding: 1rem; background:#fff; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }}
.segment-header {{ display:flex; gap:1rem; align-items:baseline; flex-wrap:wrap; margin-bottom:0.5rem; }}
.segment-time {{ font-family:monospace; font-size:0.8rem; color:#666; }}
.segment-speaker {{ font-weight:bold; background:#e5e5e5; padding:0.2rem 0.5rem; border-radius:4px; }}
.segment-text {{ line-height:1.5; }}
.back-link {{ margin-bottom:1rem; display:inline-block; }}
</style>
</head>
<body>
<div class="fr-container">
<div class="transcript-container">
<a href="/" class="fr-link back-link">← Retour à l'accueil</a>
<h1>{safe_file}</h1>
{temp_notice}
<div id="transcript"><pre>{html_escape(content)}</pre></div>
</div></div></body></html>"""
    )

@router.get("/status/{file_id}")
async def status_route(file_id: str):
    """Return the status of a batch job."""
    return jobs_db.get(file_id, {"status": "not_found"})

@router.get("/git_status")
async def git_status():
    """Return git status (behind count)."""
    try:
        repo = Repo(".")
        local_commit = repo.head().decode()
        try:
            porcelain.fetch(repo, "origin")
        except Exception:
            pass
        refs = repo.get_refs()
        remote_ref = b"refs/remotes/origin/main"
        if remote_ref not in refs:
            remote_ref = b"refs/remotes/origin/master"
        if remote_ref not in refs:
            return {"commit": local_commit, "behind": -1, "error": "remote ref introuvable"}
        remote_sha = refs[remote_ref].decode()
        # Count commits behind
        behind = 0
        local_obj = repo.get_object(local_commit.encode())
        remote_obj = repo.get_object(remote_sha.encode())
        while remote_obj.id != local_obj.id:
            behind += 1
            if not remote_obj.parents:
                break
            remote_obj = repo.get_object(remote_obj.parents[0])
        return {"commit": local_commit, "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}

# ---------- POST routes ----------
@router.post("/git_update")
async def git_update():
    """Pull latest git changes and restart."""
    try:
        repo = Repo(".")
        porcelain.fetch(repo, "origin")
        refs = repo.get_refs()
        remote_ref = b"refs/remotes/origin/main"
        if remote_ref not in refs:
            remote_ref = b"refs/remotes/origin/master"
        remote_sha = refs[remote_ref]
        local_ref = b"refs/heads/main"
        if local_ref not in repo.refs:
            local_ref = b"refs/heads/master"
        repo.refs[local_ref] = remote_sha
        porcelain.reset(repo, b"HEAD", b"hard")
        async def _restart():
            import sys
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        asyncio.create_task(_restart())
        return {"stdout": "Mise à jour OK, redémarrage...", "stderr": ""}
    except Exception as e:
        return {"stdout": "", "stderr": str(e)}

@router.post("/change_model")
async def change_model_route(model: str):
    """Change the ASR model at runtime."""
    global asr_engine, model_name
    model_name = model.lower()
    from backend.core.engine import SotaASR
    asr_engine = SotaASR(model_id=model_name, hf_token=os.getenv("HF_TOKEN"))
    asr_engine.load()
    return {"status": "ok"}

@router.post("/save_live_transcription/{client_id}")
async def save_live_transcription_route(client_id: str, content: str = Form(...)):
    """Save live transcription text to a clean file."""
    out_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_{ts}.txt"
    file_path = os.path.join(out_dir, filename)
    cleaned = "\n".join(
        line.lstrip()[4:] if line.lstrip().startswith("... ") else line.lstrip()
        for line in content.splitlines()
        if line.strip()
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    return {"status": "ok", "filename": filename}

@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    client_id: str = Form("anonymous"),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
):
    """Receive a chunk of a batch upload and trigger processing when complete."""
    if chunk_index == 0:
        # Create a new job entry (FIFO)
        add_job(file_id, {"status": "uploading", "progress": 0})
    upload_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(upload_dir, exist_ok=True)
    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())
    if chunk_index == total_chunks - 1:
        # All chunks received – assemble and process
        _update_job_status(file_id, "processing:Réassemblage...", 0)
        assembled_path = os.path.join(upload_dir, "audio_full.wav")
        with open(assembled_path, "wb") as out_f:
            for i in range(total_chunks):
                part = os.path.join(upload_dir, f"chunk_{i:04d}")
                with open(part, "rb") as in_f:
                    out_f.write(in_f.read())
                os.remove(part)
        background_tasks.add_task(_gpu_job, assembled_path, file_id, client_id)
    return {"status": "ok"}

# ---------- Background processing ----------
async def _gpu_job(assembled_path: str, file_id: str, client_id: str):
    """Process a batch audio file, store transcript, and update job status."""
    # Update status to indicate transcription start
    _update_job_status(file_id, "processing:Transcription...", 10)
    # Ensure the ASR engine is loaded
    if not getattr(asr_engine, "_loaded", False):
        asr_engine.load()
    # Progress callback updates job status
    def progress_callback(step: str, pct: int):
        pct = max(0, min(100, pct))
        _update_job_status(file_id, f"processing:{step}", pct)
    # Run the ASR processing pipeline
    transcript = await asr_engine.process_file(assembled_path, progress_callback=progress_callback)
    # Save transcript in client‑specific directory
    client_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(client_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_name = f"batch_{ts}.txt"
    txt_path = os.path.join(client_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    # Update job entry using add_job (replace existing entry)
    add_job(file_id, {"status": "terminé", "progress": 100, "result_file": txt_name})
    # Cleanup temporary files
    try:
        os.remove(assembled_path)
        os.rmdir(os.path.dirname(assembled_path))
    except Exception:
        pass

# ---------- WebSocket ----------
@router.websocket("/live")
async def live_endpoint(websocket: WebSocket, client_id: str = "anonymous", partial_albert: bool = False):
    """WebSocket endpoint for live transcription."""
    await websocket.accept()
    session = LiveSession(asr_engine, websocket, client_id, partial_albert=partial_albert)
    processor = asyncio.create_task(session.process_audio_queue())
    try:
        while True:
            data = await websocket.receive_bytes()
            await session.audio_queue.put(data)
    except WebSocketDisconnect:
        pass
    finally:
        await session.audio_queue.put(None)
        await processor
        if session.full_session_audio:
            await session.save_audio_file()
            try:
                await websocket.send_json({"type": "final_done"})
            except Exception:
                pass