import os
import sys
import re
import html
import codecs
import shutil
import asyncio
import datetime

import torch
import boto3  # pyright: ignore[reportMissingImports]
from dulwich.repo import Repo  # pyright: ignore[reportMissingImports]
from dulwich import porcelain
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, RedirectResponse
from backend.config import setup_gpu, setup_warnings, TRANSCRIPTIONS_DIR, TEMP_DIR
from backend.html_ui import HTML_UI
from backend.core.engine import SotaASR
from backend.core.live import LiveSession
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI(title="SOTA ASR Server", version="4.0.0")
app.add_middleware(ProxyHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Taille maximale du dictionnaire des jobs en mémoire (évite le memory leak)
JOBS_DB_MAX_SIZE = 500

# Global state
model_name = os.getenv("ASR_MODEL", "whisper").lower()
asr_engine = SotaASR(model_id=model_name, hf_token=os.environ.get("HF_TOKEN"))
jobs_db = {}

@app.on_event("startup")
async def startup_event():
    setup_gpu()
    setup_warnings()
    asr_engine.load()

@app.get("/")
def home(request: Request):
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

@app.websocket("/live")
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

@app.get("/transcriptions")
async def list_trans(client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.exists(p): return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

@app.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(
        p,
        media_type="text/plain",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
    for subdir in ("live_audio", "batch_audio"):
        file_path = os.path.join(TRANSCRIPTIONS_DIR, subdir, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")

@app.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier transcript introuvable.")
    return FileResponse(
        file_path,
        media_type="text/plain",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, request: Request):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    safe_client_id = html.escape(client_id)
    safe_filename = html.escape(filename)
    return HTMLResponse(content=f"""
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
                            label.textContent = 'Speaker "' + speaker + '" :';
                            const input = document.createElement('input');
                            input.type = 'text';
                            input.id = 'speaker-input-' + speaker;
                            input.value = speaker;
                            input.className = 'fr-input';
                            input.dataset.currentSpeaker = speaker;
                            input.addEventListener('input', function(e) {{
                                const newName = e.target.value;
                                const oldName = e.target.dataset.currentSpeaker;
                                document.querySelectorAll('.segment-speaker[data-speaker="' + oldName + '"]').forEach(function(span) {{
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
    """)

@app.get("/status/{file_id}")
async def status_route(file_id: str):
    return jobs_db.get(file_id, {"status": "not_found"})

@app.get("/git_status")
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
            if not c.parents: break
            c = repo.get_object(c.parents[0])
        behind = 0
        c = repo.get_object(remote_commit_sha.encode())
        while True:
            if c.id.decode() in local_commits: break
            behind += 1
            if not c.parents: break
            c = repo.get_object(c.parents[0])
        return {"commit": local_commit, "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}

# Inclusion du router contenant toutes les routes POST
from backend.routes.api import router as api_router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)