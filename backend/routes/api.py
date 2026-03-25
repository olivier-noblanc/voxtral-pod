import asyncio
import os
import sys
import datetime

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
import boto3
from dulwich.repo import Repo
from dulwich import porcelain

# Import shared objects from main
from backend.main import (
    app,
    asr_engine,
    model_name,
    jobs_db,
    TRANSCRIPTIONS_DIR,
    TEMP_DIR,
    SotaASR,
    JOBS_DB_MAX_SIZE,
    format_transcription,
)

router = APIRouter()

# ---------- POST routes (déplacées depuis backend/main.py) ----------
@router.post("/git_update")
async def git_update():
    try:
        repo = Repo(".")

        porcelain.fetch(repo, "origin")

        all_refs = repo.get_refs()
        remote_branch_ref = b"refs/remotes/origin/main"
        if remote_branch_ref not in all_refs:
            remote_branch_ref = b"refs/remotes/origin/master"

        remote_sha = all_refs[remote_branch_ref]

        local_ref = b"refs/heads/main"
        if local_ref not in repo.refs:
            local_ref = b"refs/heads/master"
        repo.refs[local_ref] = remote_sha
        porcelain.reset(repo, b"HEAD", b"hard")

        async def _restart():
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(_restart())
        return {"stdout": "Mise à jour OK, redémarrage...", "stderr": ""}

    except Exception as e:
        return {"stdout": f"Erreur: {str(e)}", "stderr": ""}


@router.post("/change_model")
async def change_model_route(model: str):
    global asr_engine, model_name
    model_name = model
    asr_engine = SotaASR(model_id=model, hf_token=os.environ.get("HF_TOKEN"))
    asr_engine.load()
    return {"status": "ok"}


@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    client_id: str = Form("anonymous"),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
):
    if chunk_index == 0:
        jobs_db[file_id] = {"status": "uploading", "progress": 0}
        if len(jobs_db) > JOBS_DB_MAX_SIZE:
            oldest_keys = list(jobs_db.keys())[: len(jobs_db) - JOBS_DB_MAX_SIZE]
            for k in oldest_keys:
                jobs_db.pop(k, None)

    upload_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())

    if chunk_index == total_chunks - 1:
        jobs_db[file_id] = {"status": "processing:Réassemblage...", "progress": 0}
        assembled_path = os.path.join(upload_dir, "audio_full")
        with open(assembled_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = os.path.join(upload_dir, f"chunk_{i:04d}")
                with open(cp, "rb") as in_f:
                    out_f.write(in_f.read())
                if os.path.exists(cp):
                    os.remove(cp)

        background_tasks.add_task(run_batch_job, assembled_path, file_id, client_id)

    return {"status": "ok"}


@router.post("/cancel/{file_id}")
async def cancel_route(file_id: str):
    if file_id in jobs_db:
        jobs_db[file_id] = {"status": "cancelled"}
    return {"status": "ok"}


@router.post("/save_live_transcription/{client_id}")
async def save_live_transcription_route(client_id: str, content: str = Form(...)):
    out_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_{ts}.txt"
    file_path = os.path.join(out_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "filename": filename}


@router.post("/update_transcription/{client_id}/{filename}")
async def update_transcription_route(client_id: str, filename: str, content: str = Form(...)):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return RedirectResponse(url=f"/view/{client_id}/{filename}", status_code=303)


@router.post("/upload_s3")
async def upload_s3_route(
    filename: str = Form(...),
    content: str = Form(...),
    endpoint: str = Form(...),
    bucket: str = Form(...),
    access_key: str = Form(...),
    secret_key: str = Form(...),
):
    if boto3 is None:
        raise HTTPException(status_code=500, detail="Le module boto3 n'est pas installé.")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        s3.put_object(
            Bucket=bucket,
            Key=filename,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- GET / WebSocket routes (déplacées depuis backend/main.py) ----------
@router.get("/")
async def home(request: Request):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    no_gpu = "true" if device == "cpu" else "false"
    try:
        repo = Repo(".")
        commit = repo.head().decode()
        print(f"[DEBUG] home() – commit envoyé au client : {commit}")
    except Exception as e:
        print(f"[ERROR] home() – erreur lors de la récupération du commit : {e}")
        commit = "unknown"
    protocol = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host")
    ws_url = f"{protocol}://{host}/live?client_id={{getClientId()}}&partial_albert={{partialAlbert}}"
    return HTMLResponse(
        content=HTML_UI
        .replace("{{model_name}}", model_name)
        .replace("{{device}}", device)
        .replace("{{no_gpu}}", no_gpu)
        .replace("{{commit}}", commit)
        .replace("{{ws_url}}", ws_url)
    )


@router.websocket("/live")
async def live_endpoint(websocket: WebSocket, client_id: str = "anonymous", partial_albert: bool = False):
    await websocket.accept()
    session = LiveSession(asr_engine, websocket, client_id, partial_albert=partial_albert)
    processor_task = asyncio.create_task(session.process_audio_queue())

    try:
        while True:
            data = await websocket.receive_bytes()
            await session.audio_queue.put(data)
    except WebSocketDisconnect:
        print(f"[*] [{client_id}] Live client disconnected.")
    finally:
        await session.audio_queue.put(None)
        await processor_task
        if session.full_session_audio:
            await session.save_audio_file()


@router.get("/transcriptions")
async def list_trans(client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.exists(p):
        return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)


@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(
        p,
        media_type="text/plain",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
    for subdir in ("live_audio", "batch_audio"):
        file_path = os.path.join(TRANSCRIPTIONS_DIR, subdir, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")


@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier transcript introuvable.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, request: Request):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    safe_client_id = html.escape(client_id)
    safe_filename = html.escape(filename)
    return HTMLResponse(
        content=f\"\"\"
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{safe_filename} – Voxtral Pod</title>
            <link rel="stylesheet" href="/static/dsfr.min.css">
            <style>
                body {{ background: var(--background-alt-grey); }}
                .transcript-container {{ max-width: 900px; margin: 2rem auto; padding: 1rem; }}
                .segment {{ margin-bottom: 1.5rem; padding: 1rem; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .segment-header {{ display: flex; gap: 1rem; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.5rem; }}
                .segment-time {{ font-family: monospace; font-size: 0.8rem; color: #666; }}
                .segment-speaker {{ font-weight: bold; background: #e5e5e5; padding: 0.2rem 0.5rem; border-radius: 4px; }}
                .segment-text {{ line-height: 1.5; }}
                .back-link {{ margin-bottom: 1rem; display: inline-block; }}
            </style>
        </head>
        <body>
            <div class="fr-container">
                <div class="transcript-container">
                    <a href="/" class="fr-link back-link">← Retour à l'accueil</a>
                    <h1>{safe_filename}</h1>
                    <button class="fr-btn fr-btn--secondary" onclick="toggleSpeakerEditor()">Modifier les speakers</button>
                    <div id="speakerRenameContainer" style="display:none;" class="fr-mt-2w"><div id="speakerRenameList" class="fr-grid-row fr-grid-row--gutters"></div></div>
                    <div id="transcript">
                        {format_transcription(content)}
                    </div>
                    <form id="updateForm" method="post" action="/update_transcription/{safe_client_id}/{safe_filename}">
                        <input type="hidden" name="content" id="updatedContentInput" value="">
                        <button type="submit" class="fr-btn fr-btn--primary">Enregistrer les modifications</button>
                    </form>
                    <script>
                    function toggleSpeakerEditor() {{
                        const container = document.getElementById('speakerRenameContainer');
                        if (container.style.display === 'none') {{
                            container.style.display = 'block';
                            initSpeakerEditor();
                        }} else {{
                            container.style.display = 'none';
                        }}
                    }}
                    function initSpeakerEditor() {{
                        const speakerSpans = document.querySelectorAll('.segment-speaker');
                        const speakers = new Set();
                        speakerSpans.forEach(function(span) {{
                            const speaker = span.dataset.speaker;
                            if (speaker) speakers.add(speaker);
                        }});
                        const listDiv = document.getElementById('speakerRenameList');
                        listDiv.innerHTML = '';
                        speakers.forEach(function(speaker) {{
                            const colDiv = document.createElement('div');
                            colDiv.className = 'fr-col-12 fr-col-md-6';
                            const label = document.createElement('label');
                            label.htmlFor = 'speaker-input-' + speaker;
                            label.textContent = 'Speaker \"' + speaker + '\" :';
                            const input = document.createElement('input');
                            input.type = 'text';
                            input.id = 'speaker-input-' + speaker;
                            input.value = speaker;
                            input.className = 'fr-input';
                            input.dataset.currentSpeaker = speaker;
                            input.addEventListener('input', function(e) {{
                                const newName = e.target.value;
                                const oldName = e.target.dataset.currentSpeaker;
                                document.querySelectorAll('.segment-speaker[data-speaker=\"' + oldName + '\"]')
                                    .forEach(function(span) {{
                                        span.textContent = newName;
                                        span.dataset.speaker = newName;
                                    }});
                                e.target.dataset.currentSpeaker = newName;
                            }});
                            colDiv.appendChild(label);
                            colDiv.appendChild(input);
                            listDiv.appendChild(colDiv);
                        }});
                    }}
                    function getUpdatedContent() {{
                        const segments = document.querySelectorAll('.segment');
                        const lines = [];
                        segments.forEach(function(seg) {{
                            const timeElem = seg.querySelector('.segment-time');
                            const time = timeElem ? timeElem.textContent.trim() : '';
                            const speakerElem = seg.querySelector('.segment-speaker');
                            const speaker = speakerElem ? speakerElem.textContent.trim() : '';
                            const textElem = seg.querySelector('.segment-text');
                            const text = textElem ? textElem.textContent.trim() : '';
                            if (speaker) {{
                                if (time) {{
                                    lines.push('[' + time + '] [' + speaker + '] ' + text);
                                }} else {{
                                    lines.push('[' + speaker + '] ' + text);
                                }}
                            }} else if (text) {{
                                lines.push(text);
                            }}
                        }});
                        return lines.join('\\n');
                    }}
                    document.getElementById('updateForm').addEventListener('submit', function(e) {{
                        const updated = getUpdatedContent();
                        document.getElementById('updatedContentInput').value = updated;
                    }});
                    </script>
                </div>
            </div>
        </body>
        </html>
        \"\"\"
    )


@router.get("/status/{file_id}")
async def status_route(file_id: str):
    return jobs_db.get(file_id, {"status": "not_found"})


@router.get("/git_status")
async def git_status():
    try:
        repo = Repo(".")
        local_commit = repo.head().decode()
        try:
            porcelain.fetch(repo, "origin")
        except Exception as e:
            print(f"[WARNING] fetch failed: {e}")
        all_refs = repo.get_refs()
        remote_branch_ref = b"refs/remotes/origin/main"
        if remote_branch_ref not in all_refs:
            remote_branch_ref = b"refs/remotes/origin/master"
        if remote_branch_ref not in all_refs:
            return {"commit": local_commit, "behind": -1, "error": "remote ref introuvable"}
        remote_commit_sha = all_refs[remote_branch_ref].decode()
        local_commits = set()
        c = repo.get_object(local_commit.encode())
        while True:
            local_commits.add(c.id.decode())
            if not c.parents:
                break
            c = repo.get_object(c.parents[0])
        behind = 0
        c = repo.get_object(remote_commit_sha.encode())
        while True:
            if c.id.decode() in local_commits:
                break
            behind += 1
            if not c.parents:
                break
            c = repo.get_object(c.parents[0])
        return {"commit": local_commit, "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}