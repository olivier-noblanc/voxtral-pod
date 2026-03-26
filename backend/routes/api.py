import os
import pathlib
import datetime
import asyncio
import shutil
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from html import escape as html_escape
from backend.html_ui import HTML_UI

# Import shared state (no import of server_asr_ref)
from backend.state import get_job, update_job, add_job, JOBS_DB_MAX_SIZE, get_asr_engine, get_current_model, set_current_model
from backend.config import TEMP_DIR, TRANSCRIPTIONS_DIR
from backend.utils import format_transcription

# Lazy import of speaker_profiles to avoid eager module loading during test collection.
# This prevents the module from being cached before environment variables (e.g., SPEAKER_PROFILES_DB)
# are set in tests that import this API module.
# The actual import is performed inside the endpoint functions that need it.

router = APIRouter()

# ---------- Sécurité : anti path traversal ----------
def _safe_join(base: str, *parts: str) -> str:
    """
    Résout le chemin final et vérifie qu'il reste sous `base`.
    Lève HTTPException 400 en cas de tentative de path traversal.
    """
    base_resolved = pathlib.Path(base).resolve()
    target = (base_resolved / pathlib.Path(*parts)).resolve()
    if not str(target).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Chemin invalide.")
    return str(target)

def _validate_client_id(client_id: str):
    """Vérifie que le client_id est valide (format user_xxxxxxxx)."""
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_") or len(client_id) < 6:
        raise HTTPException(status_code=400, detail="Identifiant client (UUID) invalide ou anonyme non autorisé.")

# ---------- Helper ----------
def _update_job_status(job_id: str, status: str, progress: int = 0, result_file: str | None = None):
    """Update a job entry in the SQLite state."""
    data = {"status": status, "progress": progress}
    if result_file:
        data["result_file"] = result_file
    update_job(job_id, data)

# ---------- GET routes ----------
@router.get("/")
async def home(request: Request):
    """Render the main HTML UI."""
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:7]
    except (Exception, ImportError):
        commit = "unknown"
    protocol = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host")
    current_model = get_current_model()
    # Force mock if TESTING is set
    if os.getenv("TESTING") == "1":
        current_model = "mock"
    engine = get_asr_engine(load_model=False)
    no_gpu_str = str(engine.no_gpu).lower()
    ws_url = f"{protocol}://{host}/live?client_id={{getClientId()}}&partial_albert={no_gpu_str}"
    return HTMLResponse(
        content=HTML_UI
        .replace("{{model_name}}", current_model)
        .replace("{{device}}", "cuda" if not engine.no_gpu else "cpu")
        .replace("{{commit}}", commit)
        .replace("{{ws_url}}", ws_url)
    )

@router.get("/transcriptions")
async def list_trans(client_id: str):
    """List transcription files for a client."""
    base = pathlib.Path(TRANSCRIPTIONS_DIR).resolve()
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.isdir(p):
        return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    """Return a transcription file."""
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type="text/plain", filename=filename)

@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
    """Download a WAV audio file."""
    # Strict isolation: filename must contain the client_id to prevent cross-client access
    if f"_{client_id}_" not in filename:
        raise HTTPException(status_code=403, detail="Accès refusé : ce fichier n'appartient pas à votre session.")

    for subdir in ("live_audio", "batch_audio"):
        try:
            file_path = _safe_join(TRANSCRIPTIONS_DIR, subdir, filename)
            if os.path.isfile(file_path):
                return FileResponse(file_path, media_type="audio/wav", filename=filename)
        except HTTPException:
            # Si _safe_join lève une erreur pour un sous-répertoire, on continue
            continue
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")

@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
    """Download a transcript text file."""
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fichier transcript introuvable.")
    with open(file_path, "rb") as f:
        content = f.read()
    return Response(
        content,
        media_type="text/plain",
        headers={
            "Content-Type": "text/plain",
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )

@router.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, request: Request):
    """Render a transcription in HTML."""
    client_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if os.path.isfile(client_path):
        file_path = client_path
        is_temp = False
    else:
        # Check live_audio (temporary files)
        temp_path = _safe_join(TRANSCRIPTIONS_DIR, "live_audio", filename)
        if os.path.isfile(temp_path):
            # Strict isolation for temp files too
            if f"_user_" in filename and f"_{client_id}_" not in filename:
                raise HTTPException(status_code=403, detail="Accès refusé.")
            file_path = temp_path
            is_temp = True
        else:
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse and format each line
    segments_html = "\n".join(format_transcription(line) for line in content.splitlines() if line.strip())

    safe_file = html_escape(filename)
    temp_notice = "<p class=\"fr-text--sm fr-mb-2w\"><em>Version temporaire (en cours de nettoyage)</em></p>" if is_temp else ""
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="fr" data-fr-scheme="dark">
<head>
<meta charset="UTF-8"><title>{safe_file} – Voxtral Pod</title>
<link rel="stylesheet" href="/static/dsfr.min.css">
<style>
body {{ background: var(--background-alt-grey); padding-bottom: 4rem; }}
.transcript-container {{ max-width: 900px; margin: 2rem auto; padding: 1rem; }}
.segment {{ margin-bottom: 1.5rem; padding: 1rem; background: var(--background-default-grey); border-radius:8px; border: 1px solid var(--border-default-grey); }}
.segment-header {{ display:flex; gap:1rem; align-items:baseline; flex-wrap:wrap; margin-bottom:0.5rem; border-bottom: 1px solid var(--border-subtle-grey); padding-bottom: 0.2rem; }}
.segment-time {{ font-family:monospace; font-size:0.8rem; color: var(--text-mention-grey); }}
.segment-speaker {{ font-weight:bold; background: var(--background-contrast-info); color: var(--text-label-info); padding:0.1rem 0.4rem; border-radius:4px; font-size: 0.9rem; }}
.segment-text {{ line-height:1.6; font-size: 1.1rem; }}
.back-link {{ margin-bottom:1rem; display:inline-block; }}
#speakerRenameContainer {{ background: var(--background-contrast-info); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-default-info); }}

/* Masquage dynamique pour copier-coller LLM */
#transcript.hide-timestamps .segment-time {{ display: none !important; }}
#transcript.hide-speakers .segment-speaker {{ display: none !important; }}
#transcript.hide-timestamps.hide-speakers .segment-header {{ display: none !important; }}
</style>
</head>
<body>
<div class="fr-container">
<div class="transcript-container">
<a href="/" class="fr-link back-link">← Retour à l'accueil</a>
<h1>{safe_file}</h1>
{temp_notice}

<div class="fr-grid-row fr-grid-row--gutters fr-mb-4w">
    <div class="fr-col-12 fr-col-md-6">
        <button id="toggleSpeakerEditorBtn" class="fr-btn fr-btn--secondary fr-btn--icon-left fr-icon-user-line fr-w-100">👥 Éditer les speakers</button>
    </div>
    <div class="fr-col-12 fr-col-md-6">
        <div class="fr-fieldset fr-p-2w" style="background: var(--background-alt-blue-france); border-radius: 8px;">
            <p class="fr-text--xs fr-mb-1w fr-text--bold">Options Copie LLM (ChatGPT/Claude)</p>
            <div class="fr-checkbox-group fr-checkbox-group--sm">
                <input type="checkbox" id="hideTimestamps">
                <label class="fr-label" for="hideTimestamps">Masquer les timestamps</label>
            </div>
            <div class="fr-checkbox-group fr-checkbox-group--sm">
                <input type="checkbox" id="hideSpeakers">
                <label class="fr-label" for="hideSpeakers">Masquer les speakers</label>
            </div>
        </div>
    </div>
</div>

<div id="speakerRenameContainer" style="display:none;" class="fr-mb-4w">
    <h3 class="fr-h6">Renommer les intervenants</h3>
    <div id="speakerRenameList" class="fr-grid-row fr-grid-row--gutters"></div>
</div>

<div id="transcript">
{segments_html}
</div>

</div></div>

<script>
// Gestion du masquage des métadonnées
document.getElementById('hideTimestamps').addEventListener('change', e => {{
    document.getElementById('transcript').classList.toggle('hide-timestamps', e.target.checked);
}});
document.getElementById('hideSpeakers').addEventListener('change', e => {{
    document.getElementById('transcript').classList.toggle('hide-speakers', e.target.checked);
}});

function toggleSpeakerEditor() {{
    const container = document.getElementById('speakerRenameContainer');
    if (!container) return;
    if (container.style.display === 'none' || container.style.display === '') {{
        container.style.display = 'block';
        initSpeakerEditor();
    }} else {{
        container.style.display = 'none';
    }}
}}

function initSpeakerEditor() {{
    const speakerSpans = document.querySelectorAll('.segment-speaker');
    const speakers = new Set();
    speakerSpans.forEach(span => {{
        const speaker = span.dataset.speaker;
        if (speaker) speakers.add(speaker);
    }});
    const listDiv = document.getElementById('speakerRenameList');
    listDiv.innerHTML = '';
    
    if (speakers.size === 0) {{
        listDiv.innerHTML = '<div class="fr-col-12"><p class="fr-text--sm">Aucun speaker détecté dans cette transcription.</p></div>';
        return;
    }}

    speakers.forEach(speaker => {{
        const colDiv = document.createElement('div');
        colDiv.className = 'fr-col-12 fr-col-md-6';
        
        const formGroup = document.createElement('div');
        formGroup.className = 'fr-input-group';
        
        const label = document.createElement('label');
        label.className = 'fr-label';
        label.htmlFor = 'speaker-input-' + speaker;
        label.textContent = 'Nom pour "' + speaker + '" :';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.id = 'speaker-input-' + speaker;
        input.value = speaker;
        input.className = 'fr-input';
        input.dataset.originalSpeaker = speaker;
        
        input.addEventListener('input', e => {{
            const newName = e.target.value;
            const originalSpeaker = e.target.dataset.originalSpeaker;
            document.querySelectorAll('.segment-speaker[data-speaker="' + originalSpeaker + '"]').forEach(span => {{
                span.textContent = newName;
            }});
        }});
        
        formGroup.appendChild(label);
        formGroup.appendChild(input);
        colDiv.appendChild(formGroup);
        listDiv.appendChild(colDiv);
    }});
}}

document.getElementById('toggleSpeakerEditorBtn').addEventListener('click', toggleSpeakerEditor);
</script>

</body></html>"""
    )

@router.get("/status/{file_id}")
async def status_route(file_id: str):
    """Return the status of a batch job."""
    return get_job(file_id, {"status": "not_found"})

@router.get("/git_status")
async def git_status():
    """Return git status (behind count) using standard git CLI."""
    try:
        import subprocess
        # Fetch quietly
        subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, check=False)
        
        # Current SHA
        local_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        
        # Behind count
        behind_out = subprocess.check_output(["git", "rev-list", "--count", "HEAD..origin/main"], text=True).strip()
        behind = int(behind_out) if behind_out.isdigit() else 0
        
        return {"commit": local_sha[:7], "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}

# ---------- Speaker profile endpoints ----------
@router.get("/speaker_profiles")
async def get_speaker_profiles():
    """
    Return the list of stored speaker profiles.
    Each item is a dict: { "speaker_id": "...", "name": "..." }
    """
    # Import here to ensure the latest environment configuration is respected.
    import backend.core.speaker_profiles as speaker_profiles
    profiles = speaker_profiles.load_profiles()
    # Convert to list of dicts, ignore embeddings for the API response
    return [
        {"speaker_id": sid, "name": name}
        for sid, (name, _) in profiles.items()
    ]

@router.post("/speaker_profiles")
async def upsert_speaker_profile(payload: dict):
    # Import here to ensure the latest environment configuration is respected.
    import backend.core.speaker_profiles as speaker_profiles
    """
    Create or update a speaker profile.
    Expected JSON payload:
    {
        "speaker_id": "SPEAKER_00",
        "name": "Olivier",
        "embedding": [0.12, 0.34, ...]   # optional, list of floats
    }
    """
    speaker_id = payload.get("speaker_id")
    name = payload.get("name")
    embedding = payload.get("embedding")
    if not speaker_id or not name:
        raise HTTPException(status_code=400, detail="speaker_id and name are required.")
    # Convert embedding list to numpy array if provided
    emb_array = None
    if embedding is not None:
        try:
            import numpy as np
            emb_array = np.array(embedding, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding format.")
    speaker_profiles.save_profile(speaker_id, name, emb_array)
    return {"status": "ok"}

@router.post("/segment_update")
async def update_segment_speaker(payload: dict):
    """
    Update the speaker label of a specific segment in a transcription.
    Expected payload:
    {
        "client_id": "user_abcdef12",
        "filename": "live_20240101_123456.txt",
        "segment_index": 3,          # zero‑based index in the transcript file
        "new_speaker": "Julie"
    }
    The server rewrites the transcript file with the new speaker label.
    """
    client_id = payload.get("client_id")
    filename = payload.get("filename")
    segment_index = payload.get("segment_index")
    new_speaker = payload.get("new_speaker")
    if not all([client_id, filename, isinstance(segment_index, int), new_speaker]):
        raise HTTPException(status_code=400, detail="Invalid payload.")
    # Load the transcript file
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Simple format: each line is "start end speaker text"
    # We replace the third token (speaker) for the given index
    if segment_index < 0 or segment_index >= len(lines):
        raise HTTPException(status_code=400, detail="segment_index out of range.")
    parts = lines[segment_index].split(maxsplit=2)
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Malformed transcription line.")
    # Rebuild line with new speaker
    parts[2] = new_speaker
    lines[segment_index] = " ".join(parts) + "\n"
    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return {"status": "ok"}

# ---------- POST routes ----------
@router.post("/git_update")
async def git_update():
    """Pull & Reset using Git CLI and restart the container."""
    try:
        import subprocess
        # Force sync with origin/main (consistent with run.sh)
        subprocess.run(["git", "fetch", "origin", "main"], check=True)
        res = subprocess.run(["git", "reset", "--hard", "origin/main"], capture_output=True, text=True, check=True)
        
        async def _restart():
            await asyncio.sleep(2)
            print("[INFO] Git CLI update OK. Exiting container...")
            os._exit(0)
            
        asyncio.create_task(_restart())
        return {"stdout": res.stdout or "Mise à jour réussie. Redémarrage...", "stderr": ""}
    except Exception as e:
        return {"stdout": "", "stderr": str(e)}

@router.post("/change_model")
async def change_model_route(model: str):
    """Change the ASR model at runtime."""
    set_current_model(model.lower())
    # Pre-warm the model in this worker
    get_asr_engine(load_model=True)
    return {"status": "ok"}


@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    client_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
):
    """Receive a chunk of a batch upload and trigger processing when complete."""
    _validate_client_id(client_id)
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
        await asyncio.to_thread(_assemble_chunks, assembled_path, total_chunks, upload_dir)
        background_tasks.add_task(_gpu_job, assembled_path, file_id, client_id)
    return {"status": "ok"}

# ---------- Helper I/O (sync, appelé via asyncio.to_thread) ----------
def _assemble_chunks(assembled_path: str, total_chunks: int, upload_dir: str) -> None:
    """Concatène les chunks en un seul fichier, puis les supprime."""
    with open(assembled_path, "wb") as out_f:
        for i in range(total_chunks):
            part = os.path.join(upload_dir, f"chunk_{i:04d}")
            with open(part, "rb") as in_f:
                out_f.write(in_f.read())
            os.remove(part)

async def _gpu_job(assembled_path: str, file_id: str, client_id: str):
    """Process a batch audio file, store transcript, and update job status."""
    print(f"[*] Starting GPU Job for {file_id} (Client: {client_id})")
    # Update status to indicate transcription start
    _update_job_status(file_id, "processing:Transcription...", 10)
    # Ensure the ASR engine is loaded
    engine = get_asr_engine(load_model=True)
    # Progress callback updates job status
    def progress_callback(step: str, pct: int):
        pct = max(0, min(100, pct))
        print(f"[*] [Batch {file_id}] {step} : {pct}%")
        _update_job_status(file_id, f"processing:{step}", pct)
    # Run the ASR processing pipeline
    transcript = await engine.process_file(assembled_path, progress_callback=progress_callback)
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
    
    # Save audio for history before cleanup
    try:
        batch_audio_dir = os.path.join(TRANSCRIPTIONS_DIR, "batch_audio")
        os.makedirs(batch_audio_dir, exist_ok=True)
        # Expected filename format for frontend: batch_{client_id}_{ts}.wav
        # Wait, the frontend code (loadHistory) says:
        # audioFilename = 'batch_' + getClientId() + '_' + ts + '.wav';
        # where ts = f.replace('batch_', '').replace('.txt', '');
        # So we use the same 'ts' as the txt file.
        archived_audio_name = f"batch_{client_id}_{ts}.wav"
        archived_audio_path = os.path.join(batch_audio_dir, archived_audio_name)
        shutil.copy2(assembled_path, archived_audio_path)
    except Exception:
        pass

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
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_"):
        # On ne peut pas lever HTTPException ici, on ferme la socket
        await websocket.close(code=4003)
        return
    await websocket.accept()
    from backend.core.live import LiveSession
    engine = get_asr_engine(load_model=True)
    session = LiveSession(engine, websocket, client_id, partial_albert=partial_albert)
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
            # Lancement de la transcription finale de manière asynchrone
            # pour qu'elle puisse se terminer même si la socket est coupée
            asyncio.create_task(session.save_audio_file())
