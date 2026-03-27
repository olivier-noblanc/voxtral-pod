import os
import pathlib
import datetime
import asyncio
import shutil
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from html import escape as html_escape
from backend.html_ui import HTML_UI
import markdown

# Import shared state (no import of server_asr_ref)
from backend.state import get_job, update_job, add_job, JOBS_DB_MAX_SIZE, get_asr_engine, get_current_model, set_current_model
from backend.config import TEMP_DIR, TRANSCRIPTIONS_DIR, BASE_DIR
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

def _require_admin_key(request: Request):
    """
    Vérifie la clé admin dans le header X-Admin-Key.
    Si ADMIN_API_KEY n'est pas définie dans l'environnement, l'accès est libre (rétro-compat).
    Si elle est définie, le header doit correspondre exactement.
    """
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        return  # Pas de clé configurée → accès libre
    provided = request.headers.get("X-Admin-Key", "")
    if provided != admin_key:
        raise HTTPException(status_code=403, detail="Clé admin invalide ou manquante.")

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
    ws_url = f"{protocol}://{host}/live?client_id={{{{getClientId()}}}}&partial_albert={no_gpu_str}"
    # Build model options based on device availability
    if engine.no_gpu:
        model_options = '<option value="albert" selected>API Albert (Étalab)</option>'
    else:
        model_options = (
            '<option value="whisper">Faster-Whisper Large-v3</option>'
            '<option value="voxtral">Voxtral Mini 4B</option>'
            '<option value="albert">API Albert (Étalab)</option>'
            '<option value="mock">MODE TEST (Mock ASR)</option>'
        )
    return HTMLResponse(
        content=HTML_UI
        .replace("{{model_name}}", current_model)
        .replace("{{device}}", "cuda" if not engine.no_gpu else "cpu")
        .replace("{{commit}}", commit)
        .replace("{{ws_url}}", ws_url)
        .replace("{{model_options}}", model_options)
    )

@router.get("/transcriptions")
async def list_trans(client_id: str):
    """List transcription files for a client."""
    base = pathlib.Path(TRANSCRIPTIONS_DIR).resolve()
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.isdir(p):
        return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

# Alias for /transcriptions with trailing slash to satisfy route contract tests.
@router.get("/transcriptions/")
async def list_trans_slash(client_id: str):
    """Alias for /transcriptions with trailing slash."""
    return await list_trans(client_id)

@router.delete("/transcriptions/{filename}")
async def delete_trans(filename: str, client_id: str):
    """Supprime un fichier de transcription et son audio associé."""
    # Supprimer le fichier de transcription
    trans_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(trans_path):
        raise HTTPException(status_code=404, detail="Fichier transcription introuvable.")
    os.remove(trans_path)

    # Supprimer le fichier audio associé s'il existe (live ou batch)
    # Le nom d'audio suit le même schéma dans la vue
    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_name = f"batch_{client_id}_{ts}.wav"
        audio_dir = os.path.join(TRANSCRIPTIONS_DIR, "batch_audio")
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_name = f"live_{client_id}_{ts}.wav"
        audio_dir = os.path.join(TRANSCRIPTIONS_DIR, "live_audio")
    audio_path = os.path.join(audio_dir, audio_name)
    if os.path.isfile(audio_path):
        os.remove(audio_path)

    return {"status": "ok"}

@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    """Return a transcription file."""
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type="text/plain", filename=filename)

@router.post("/summary/{filename}")
async def generate_summary(filename: str, client_id: str):
    """Générer un compte-rendu avec l'API Albert depuis une transcription."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    from backend.core.postprocess import summarize_text
    try:
        summary = await asyncio.to_thread(summarize_text, content)
        return {"status": "ok", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actions/{filename}")
async def generate_actions(filename: str, client_id: str):
    """Extraire les décisions et TODOs avec l'API Albert depuis une transcription."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    from backend.core.postprocess import extract_actions_text
    try:
        actions_list = await asyncio.to_thread(extract_actions_text, content)
        actions_text = "\n".join(f"- {a}" for a in actions_list)
        return {"status": "ok", "actions": actions_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup/{filename}")
async def generate_cleanup(filename: str, client_id: str):
    """Nettoyer les tics de langage d'une transcription avec l'API Albert."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    from backend.core.postprocess import clean_text
    try:
        cleaned = await asyncio.to_thread(clean_text, content)
        return {"status": "ok", "cleanup": cleaned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
    """Download a WAV audio file from the local filesystem."""
    # Strict isolation: filename must contain the client_id to prevent cross-client access
    if f"_{client_id}_" not in filename:
        raise HTTPException(status_code=403, detail="Accès refusé : ce fichier n'appartient pas à votre session.")

    for subdir in ("live_audio", "batch_audio"):
        try:
            file_path = _safe_join(TRANSCRIPTIONS_DIR, subdir, filename)
            if os.path.isfile(file_path):
                return FileResponse(
                    file_path,
                    media_type="audio/wav",
                    filename=filename,
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        except HTTPException:
            # Si _safe_join lève une erreur pour un sous‑répertoire, on continue
            continue
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")

@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
    """Download a transcript text file from the local filesystem."""
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
    """Render une transcription en HTML avec synchronisation audio et waveform."""
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

    # Parse et formate chaque ligne
    segments_html = "\n".join(format_transcription(line) for line in content.splitlines() if line.strip())

    safe_file = html_escape(filename)
    temp_notice = "<p class=\"fr-text--sm fr-mb-2w\"><em>Version temporaire (en cours de nettoyage)</em></p>" if is_temp else ""

    # Détermine le nom du fichier audio pour la lecture
    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_filename = f"batch_{client_id}_{ts}.wav"
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_filename = f"live_{client_id}_{ts}.wav"
    audio_url = f"/download_audio/{client_id}/{audio_filename}"

    html = f"""<!DOCTYPE html>
<html lang="fr" data-fr-scheme="dark">
<head>
<meta charset="UTF-8"><title>{safe_file} – Voxtral Pod</title>
<link rel="stylesheet" href="/static/dsfr.min.css">
</head>
<body>
<div class="fr-container">
<div class="transcript-container">
<a href="/" class="fr-link back-link">← Retour à l'accueil</a>
<h1>{safe_file}</h1>
{temp_notice}
<div class="audio-player-container">
    <audio id="audioPlayer" controls style="width:100%; margin-bottom:1rem;"></audio>
    <div id="waveform"></div>
</div>
<div class="fr-grid-row fr-grid-row--gutters fr-mb-4w">
    <div class="fr-col-12 fr-col-md-6">
        <button id="toggleSpeakerEditorBtn" class="fr-btn fr-btn--secondary fr-btn--icon-left fr-icon-user-line fr-w-100">👥 Éditer les speakers</button>
    </div>
    <div class="fr-col-12 fr-col-md-6">
            <div class="fr-fieldset fr-p-2w" style="background: var(--background-alt-blue-france); border-radius:8px;">
            <p class="fr-text--xs fr-mb-1w fr-text--bold">Options Copie LLM (ChatGPT/Claude)</p>
            <div class="fr-checkbox-group fr-checkbox-group--sm">
                <input type="checkbox" id="hideTimestamps"><label class="fr-label" for="hideTimestamps">Masquer les timestamps</label>
            </div>
            <div class="fr-checkbox-group fr-checkbox-group--sm">
                <input type="checkbox" id="hideSpeakers"><label class="fr-label" for="hideSpeakers">Masquer les speakers</label>
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

</body></html>
"""
    return HTMLResponse(html)

def _render_postprocess_page(title: str, content: str, filename: str) -> HTMLResponse:
    """Rendu d'une page de visualisation pour les résultats Albert (Markdown)."""
    safe_title = html_escape(title)
    safe_filename = html_escape(filename)
    safe_content = html_escape(content)
    # Rendu Markdown côté serveur
    rendered_html = markdown.markdown(content, extensions=['extra', 'nl2br'])

    html = f"""<!DOCTYPE html>
<html lang="fr" data-fr-scheme="dark">
<head>
    <meta charset="UTF-8">
    <title>{safe_title} - {safe_filename}</title>
    <link rel="stylesheet" href="/static/dsfr.min.css">
    <link rel="stylesheet" href="/static/postprocess.css">
    <script src="/static/postprocess.js"></script>
</head>
<body>
    <div class="fr-container">
        <div class="postprocess-container">
            <a href="/" class="fr-link fr-icon-arrow-left-line fr-link--icon-left fr-mb-2w">Retour à l'accueil</a>
            <h1>{safe_title}</h1>
            <p class="fr-text--sm">Fichier : {safe_filename}</p>

            <div class="actions-bar">
                <button id="copyBtn" class="fr-btn fr-btn--secondary fr-btn--icon-left fr-icon-clipboard-line">Copier le texte</button>
                <button id="downloadBtn" class="fr-btn fr-btn--secondary fr-btn--icon-left fr-icon-download-line" data-title="{safe_title}" data-filename="{safe_filename}">Télécharger (.txt)</button>
            </div>

            <div id="content" class="content-area fr-text--md" data-content="{safe_content}">
                {rendered_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(html)

@router.get("/view_summary/{client_id}/{filename}")
async def view_summary(client_id: str, filename: str):
    """Visualiser le compte-rendu Albert."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import summarize_text
    try:
        summary = await asyncio.to_thread(summarize_text, content)
        return _render_postprocess_page("Compte Rendu Albert", summary, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_actions/{client_id}/{filename}")
async def view_actions(client_id: str, filename: str):
    """Visualiser les actions Albert."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import extract_actions_text
    try:
        actions_list = await asyncio.to_thread(extract_actions_text, content)
        actions_text = "\n".join(f"- {a}" for a in actions_list)
        return _render_postprocess_page("Actions Albert", actions_text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_cleanup/{client_id}/{filename}")
async def view_cleanup(client_id: str, filename: str):
    """Visualiser le texte nettoyé Albert."""
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import clean_text
    try:
        cleaned = await asyncio.to_thread(clean_text, content)
        return _render_postprocess_page("Nettoyage Albert", cleaned, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
from typing import List, Union
@router.get("/speaker_profiles")
async def get_speaker_profiles() -> List[dict]:
    """Return the list of stored speaker profiles. Each item: { speaker_id, name }"""
    # Import here to ensure the latest environment configuration is respected.
    import backend.core.speaker_profiles as speaker_profiles
    profiles = speaker_profiles.load_profiles()
    # Convert to list of dicts, ignore embeddings for the API response
    return [
        {"speaker_id": sid, "name": name}
        for sid, (name, _) in profiles.items()
    ]

@router.post("/speaker_profiles")
async def upsert_speaker_profile(payload: dict) -> Union[dict, None]:
    """
    Create or update a speaker profile.
    Expected JSON: { "speaker_id": str, "name": str, "embedding": list[float] (optional) }
    """
    # Import here to ensure the latest environment configuration is respected.
    import backend.core.speaker_profiles as speaker_profiles
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

def _extract_and_save_profile_sync(audio_path: str, start_s: float, end_s: float, speaker_id: str):
    import os
    if not os.path.isfile(audio_path):
        print(f"[!] Audio file {audio_path} not found for embedding extraction.")
        return
    try:
        from resemblyzer import preprocess_wav, VoiceEncoder
        wav = preprocess_wav(audio_path)
        sample_rate = 16000
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        segment_audio = wav[start_idx:end_idx]
        
        if len(segment_audio) < 16000 * 0.5:
            print("[!] Audio segment too short to extract reliable voice profile.")
            return

        encoder = VoiceEncoder("cpu")
        emb = encoder.embed_utterance(segment_audio)
        import backend.core.speaker_profiles as speaker_profiles
        speaker_profiles.save_profile(speaker_id=speaker_id, name=speaker_id, embedding=emb)
        print(f"[*] Saved voice profile for speaker: {speaker_id}")
    except Exception as e:
        print(f"[!] Warning: Failed to extract speaker embedding for {speaker_id}: {e}")

@router.post("/segment_update")
async def update_segment_speaker(payload: dict, background_tasks: BackgroundTasks):
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
    import re
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
    if segment_index < 0 or segment_index >= len(lines):
        raise HTTPException(status_code=400, detail="segment_index out of range.")
    
    # Extract times for profile generation
    match = re.search(r'\[([\d.]+)s\s*->\s*([\d.]+)s\]', lines[segment_index])
    if match:
        start_s = float(match.group(1))
        end_s = float(match.group(2))
        if filename.startswith("batch_"):
            ts = filename.replace("batch_", "").replace(".txt", "")
            audio_name = f"batch_{client_id}_{ts}.wav"
            audio_dir = "batch_audio"
        else:
            ts = filename.replace("live_", "").replace(".txt", "")
            audio_name = f"live_{client_id}_{ts}.wav"
            audio_dir = "live_audio"
        audio_path = os.path.join(TRANSCRIPTIONS_DIR, audio_dir, audio_name)
        background_tasks.add_task(_extract_and_save_profile_sync, audio_path, start_s, end_s, new_speaker)

    # Format réel : [0.10s -> 1.20s] [SPEAKER_00] texte...
    # On remplace uniquement le bloc [SPEAKER_XX] en préservant le timestamp et le texte.
    new_line = re.sub(
        r'(\[[^\]]*s\s*->\s*[^\]]*s\]\s*)\[[^\]]*\]',
        rf'\1[{new_speaker}]',
        lines[segment_index]
    )
    if new_line == lines[segment_index]:
        raise HTTPException(status_code=400, detail="Format de ligne non reconnu (timestamp+speaker attendus).")
    lines[segment_index] = new_line
    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return {"status": "ok"}


# ---------- POST routes ----------
@router.post("/git_update")
async def git_update(request: Request):
    """Pull & Reset using Git CLI and restart the container."""
    _require_admin_key(request)
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
async def change_model_route(model: str, request: Request):
    """Change the ASR model at runtime."""
    _require_admin_key(request)
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
    try:
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
            archived_audio_name = f"batch_{client_id}_{ts}.wav"
            archived_audio_path = os.path.join(batch_audio_dir, archived_audio_name)
            shutil.copy2(assembled_path, archived_audio_path)
        except Exception:
            pass

    except Exception as e:
        print(f"[!] Error in GPU Job {file_id}: {e}")
        # En cas de plantage (OOM, fichier illisible, API Albert HS, etc.), informer directement le client
        from backend.state import update_job
        update_job(file_id, {"status": "erreur", "progress": 0, "error_details": str(e)})

    finally:
        # Cleanup temporary files (exécuté quoi qu'il arrive)
        try:
            if os.path.exists(assembled_path):
                os.remove(assembled_path)
            upload_dir = os.path.dirname(assembled_path)
            if os.path.exists(upload_dir) and not os.listdir(upload_dir):
                os.rmdir(upload_dir)
        except Exception:
            pass

async def _live_final_job(wav_path: str, job_id: str, client_id: str, timestamp: str):
    """Process the final live audio file as a background job using Diarization."""
    print(f"[*] Starting Live Final Job for {job_id} (Client: {client_id})")
    try:
        _update_job_status(job_id, "processing:Transcription avancée...", 5)
        engine = get_asr_engine(load_model=True)
        
        def progress_callback(step: str, pct: int):
            pct = max(0, min(100, pct))
            print(f"[*] [LiveFinal {job_id}] {step} : {pct}%")
            _update_job_status(job_id, f"processing:{step}", pct)
            
        transcript = await engine.process_file(wav_path, progress_callback=progress_callback)
        
        # Save transcript in client‑specific directory
        client_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
        os.makedirs(client_dir, exist_ok=True)
        txt_name = f"live_{timestamp}.txt"
        txt_path = os.path.join(client_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        add_job(job_id, {"status": "terminé", "progress": 100, "result_file": txt_name})
    except Exception as e:
        print(f"[!] Error in Live Final Job {job_id}: {e}")
        from backend.state import update_job
        update_job(job_id, {"status": "erreur", "progress": 0, "error_details": str(e)})

# ---------- WebSocket ----------
@router.websocket("/live")
async def live_endpoint(websocket: WebSocket, client_id: str = "anonymous", partial_albert: bool = False, device_id: str = None, session_id: str = None):
    """WebSocket endpoint for live transcription."""
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_"):
        # On ne peut pas lever HTTPException ici, on ferme la socket
        await websocket.close(code=4003)
        return
    await websocket.accept()
    from backend.core.live import LiveSession
    engine = get_asr_engine(load_model=True)
    session = LiveSession(engine, websocket, client_id, partial_albert=partial_albert, selected_audio_device_id=device_id)
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
            # Sauvegarder l'audio et déclencher le traitement asynchrone type batch
            wav_path, timestamp = await session.save_wav_only()
            if wav_path:
                job_id = session_id if session_id else f"live_final_{timestamp}"
                add_job(job_id, {"status": "en_attente", "progress": 0})
                asyncio.create_task(_live_final_job(wav_path, job_id, client_id, timestamp))