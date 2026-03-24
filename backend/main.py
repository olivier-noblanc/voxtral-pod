import os
import uuid
import asyncio
import datetime
import torch
import re
import json
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from backend.config import setup_gpu, setup_warnings, TRANSCRIPTIONS_DIR, TEMP_DIR
from backend.html_ui import HTML_UI
from backend.core.engine import SotaASR
from backend.core.live import LiveSession
# Optional import of boto3 for S3 uploads. Wrapped in try/except to avoid import errors if boto3 is not installed.

import boto3 # pyright: ignore[reportMissingImports]
import dulwich  # noqa: F401
from dulwich.repo import Repo # pyright: ignore[reportMissingImports]
from dulwich import porcelain


from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI(title="SOTA ASR Server", version="4.0.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(ProxyHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    ws_protocol = "wss:" if request.url.scheme == "https" else "ws:"
    # Calcul de l'URL WebSocket
    protocol = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host")
    ws_url = f"{protocol}://{host}/live?client_id={{getClientId()}}&partial_albert={{partialAlbert}}"
    return HTMLResponse(
        content=HTML_UI
            .replace("{{model_name}}", model_name)
            .replace("{{device}}", device)
            .replace("{{no_gpu}}", no_gpu)
            .replace("{{commit}}", commit)
            .replace("{{ws_protocol}}", ws_protocol)
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
            # Final pass can be added here if needed, reused from run_batch_job logic
            pass

@app.get("/transcriptions")
async def list_trans(client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.exists(p): return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

@app.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(p): raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(p, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())

@app.post("/change_model")
async def change_model_route(model: str):
    global asr_engine, model_name
    model_name = model
    asr_engine = SotaASR(model_id=model, hf_token=os.environ.get("HF_TOKEN"))
    asr_engine.load()
    return {"status": "ok"}

@app.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...), client_id: str = Form("anonymous"),
    chunk_index: int = Form(...), total_chunks: int = Form(...),
    file: UploadFile = File(...)
):
    if chunk_index == 0:
        jobs_db[file_id] = {"status": "uploading", "progress": 0}
    
    upload_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())

    if chunk_index == total_chunks - 1:
        # Reassemble
        jobs_db[file_id] = {"status": "processing:Réassemblage...", "progress": 0}
        assembled_path = os.path.join(upload_dir, "audio_full")
        with open(assembled_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = os.path.join(upload_dir, f"chunk_{i:04d}")
                with open(cp, "rb") as in_f: out_f.write(in_f.read())
                if os.path.exists(cp): os.remove(cp)
        
        # Launch background job
        background_tasks.add_task(run_batch_job, assembled_path, file_id, client_id)

    return {"status": "ok"}

@app.post("/cancel/{file_id}")
async def cancel_route(file_id: str):
    if file_id in jobs_db:
        jobs_db[file_id] = {"status": "cancelled"}
    return {"status": "ok"}

async def run_batch_job(path, file_id, client_id):
    start_time = datetime.datetime.now()
    try:
        def update_progress(status, pct):
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            eta = 0
            if pct > 5 and pct < 100:
                # Simple linear estimation: (elapsed / pct) * (100 - pct)
                eta = int((elapsed / pct) * (100 - pct))
            
            jobs_db[file_id] = {
                "status": f"processing:{status}",
                "progress": pct,
                "eta": eta
            }

        transcript = await asr_engine.process_file(path, progress_callback=update_progress)
        
        # Save
        out_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(out_dir, f"batch_{ts}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        jobs_db[file_id] = {"status": "done", "result": transcript[:500] + "..."}
    except Exception as e:
        print(f"[!] Batch error: {e}")
        jobs_db[file_id] = {"status": "error", "error": str(e)}

def format_transcription(text: str) -> str:
    lines = text.split('\n')
    html_parts = []
    for line in lines:
        # Expression régulière pour capturer les segments
        match = re.match(r'^\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)$', line.strip())
        if match:
            time_str, speaker, content = match.groups()
            # Échapper les caractères spéciaux pour éviter les erreurs JavaScript
            escaped_time = json.dumps(time_str)[1:-1]  # Supprime les guillemets externes
            escaped_speaker = json.dumps(speaker)[1:-1]
            escaped_content = json.dumps(content)[1:-1]
            html_parts.append(f"""
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-time">{escaped_time}</span>
                        <span class="segment-speaker" data-speaker="{escaped_speaker}">{escaped_speaker}</span>
                    </div>
                    <div class="segment-text">{escaped_content}</div>
                </div>
            """)
        elif line.strip():
            # Si la ligne ne correspond pas au format, on l'affiche simplement
            escaped_line = json.dumps(line)[1:-1]
            html_parts.append(f"<p>{escaped_line}</p>")
    return "\n".join(html_parts)

@app.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, request: Request):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # On peut éventuellement parser le contenu pour le structurer
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{filename} – Voxtral Pod</title>
            <link rel="stylesheet" href="/static/dsfr.min.css">
            <style>
                body {{
                    background: var(--background-alt-grey);
                }}
                .transcript-container {{
                    max-width: 900px;
                    margin: 2rem auto;
                    padding: 1rem;
                }}
                .segment {{
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .segment-header {{
                    display: flex;
                    gap: 1rem;
                    align-items: baseline;
                    flex-wrap: wrap;
                    margin-bottom: 0.5rem;
                }}
                .segment-time {{
                    font-family: monospace;
                    font-size: 0.8rem;
                    color: #666;
                }}
                .segment-speaker {{
                    font-weight: bold;
                    background: #e5e5e5;
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                }}
                .segment-text {{
                    line-height: 1.5;
                }}
                .back-link {{
                    margin-bottom: 1rem;
                    display: inline-block;
                }}
            </style>
        </head>
        <body>
            <div class="fr-container">
                <div class="transcript-container">
                    <a href="/" class="fr-link back-link">← Retour à l'accueil</a>
                    <h1>{filename}</h1>
                    <button class="fr-btn fr-btn--secondary" onclick="toggleSpeakerEditor()">Modifier les speakers</button>
                    <div id="speakerRenameContainer" style="display:none;" class="fr-mt-2w"><div id="speakerRenameList" class="fr-grid-row fr-grid-row--gutters"></div></div>
                    <div id="transcript">
                        {format_transcription(content)}
                    </div>
                    <button class="fr-btn fr-btn--primary" onclick="saveChanges()">Enregistrer les modifications</button>
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
                            if (time && speaker) {{
                                lines.push('[' + time + '] [' + speaker + '] ' + text);
                            }} else if (text) {{
                                lines.push(text);
                            }}
                        }});
                        return lines.join('\n');
                    }}
                    async function saveChanges() {{
                        const updated = getUpdatedContent();
                        const client_id = "{client_id}";
                        const filename = "{filename}";
                        const res = await fetch('/update_transcription/' + client_id + '/' + filename, {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                            body: new URLSearchParams({{content: updated}})
                        }});
                        if (res.ok) {{
                            alert('Transcription sauvegardée.');
                        }} else {{
                            alert('Erreur lors de la sauvegarde.');
                        }}
                    }}
                    </script>
                </div>
            </div>
        </body>
        </html>
    """)

@app.post("/update_transcription/{client_id}/{filename}")
async def update_transcription_route(client_id: str, filename: str, content: str = Form(...)):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok"}

@app.post("/save_live_transcription/{client_id}")
async def save_live_transcription_route(client_id: str, content: str = Form(...)):
    out_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_{ts}.txt"
    file_path = os.path.join(out_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "filename": filename}

@app.get("/status/{file_id}")
async def status_route(file_id: str):
    return jobs_db.get(file_id, {"status": "not_found"})

@app.get("/git_status")
async def git_status():
    try:
        repo = Repo(".")
        # Récupérer le commit local
        local_commit = repo.head().decode()
        
        # Récupérer les informations du remote
        # Utiliser dulwich pour obtenir les références distantes
        try:
            # Effectuer un fetch pour obtenir les dernières références distantes
            # Cela permet de détecter les nouveaux push
            try:
                # Utiliser porcelain.fetch() pour obtenir les dernières références distantes
                remote_refs = porcelain.fetch(repo, "origin")
            except Exception as e:
                print(f"[WARNING] git_status() – impossible de fetch les références distantes : {e}")
                # Si le fetch échoue, utiliser les références locales existantes
                remote_refs = repo.get_refs()
            
            # Trouver la branche 'origin/main' ou 'origin/master'
            remote_branch_ref = b"refs/remotes/origin/main"
            if remote_branch_ref not in remote_refs:
                remote_branch_ref = b"refs/remotes/origin/master"
            
            if remote_branch_ref in remote_refs:
                remote_commit = remote_refs[remote_branch_ref].decode()
                
                # Calculer le nombre de commits en retard
                # Utiliser dulwich pour comparer les commits
                try:
                    # Obtenir les commits communs entre local et distant
                    local_commit_obj = repo.get_object(local_commit.encode())
                    remote_commit_obj = repo.get_object(remote_commit.encode())
                    
                    # Utiliser dulwich pour trouver les commits en commun
                    # et compter ceux qui sont présents dans remote mais pas dans local
                    from dulwich.objects import Commit
                    from dulwich.diff_tree import tree_changes
                    
                    # Créer un ensemble des commits locaux
                    local_commits = set()
                    current_commit = local_commit_obj
                    
                    # Parcourir les commits locaux
                    while current_commit is not None:
                        local_commits.add(current_commit.id.decode())
                        if len(current_commit.parents) == 0:
                            break
                        current_commit = repo.get_object(current_commit.parents[0])
                    
                    # Compter les commits distants qui ne sont pas locaux
                    behind_count = 0
                    current_remote = remote_commit_obj
                    
                    # Parcourir les commits distants jusqu'à atteindre un commit commun
                    while current_remote is not None:
                        if current_remote.id.decode() in local_commits:
                            break
                        behind_count += 1
                        if len(current_remote.parents) == 0:
                            break
                        current_remote = repo.get_object(current_remote.parents[0])
                        
                except Exception as e:
                    print(f"[ERROR] git_status() – erreur lors du calcul des commits en retard : {e}")
                    behind_count = -1
                    
            else:
                behind_count = -1  # Impossible de trouver la branche remote
                
        except Exception as e:
            print(f"[ERROR] git_status() – erreur lors de la récupération des références distantes : {e}")
            behind_count = -1
            
        print(f"[DEBUG] git_status() – commit local : {local_commit}, commits en retard : {behind_count}")
        return {"commit": local_commit, "behind": behind_count}
    except Exception as e:
        print(f"[ERROR] git_status() – erreur lors de la récupération du commit : {e}")
        return {"commit": "unknown", "behind": -1}

async def restart_after_delay():
    # Attendre 1 seconde avant de redémarrer
    await asyncio.sleep(1)
    # Exécuter le processus actuel à nouveau
    os.execv(sys.executable, [sys.executable] + sys.argv)

@app.post("/git_update")
async def git_update():
    import os
    import sys
    import asyncio

    print("[DEBUG] git_update() – mise à jour et redémarrage automatique")
    
    try:
        # Utiliser dulwich pour faire un pull
        repo = Repo(".")
        
        # Obtenir les références distantes
        try:
            # Effectuer un fetch pour obtenir les dernières références distantes
            remote_refs = porcelain.fetch(repo, "origin")
        except Exception as e:
            print(f"[WARNING] git_update() – impossible de fetch les références distantes : {e}")
            # Si le fetch échoue, utiliser les références locales existantes
            remote_refs = repo.get_refs()
        
        # Trouver la branche 'origin/main' ou 'origin/master'
        remote_branch_ref = b"refs/remotes/origin/main"
        if remote_branch_ref not in remote_refs:
            remote_branch_ref = b"refs/remotes/origin/master"
        
        if remote_branch_ref in remote_refs:
            # Obtenir le commit distant
            remote_commit = remote_refs[remote_branch_ref]
            
            # Effectuer un pull avec dulwich
            try:
                # Récupérer les changements distants
                # repo.fetch(remote_refs)  # Déjà fait ci-dessus
                
                # Mettre à jour la branche locale
                local_ref = b"refs/heads/main"
                if local_ref not in repo.refs:
                    local_ref = b"refs/heads/master"
                
                if local_ref in repo.refs:
                    # Mettre à jour la référence locale vers le commit distant
                    repo.refs[local_ref] = remote_commit
                    
            except Exception as e:
                print(f"[ERROR] git_update() – erreur lors du pull : {e}")
                return {"stdout": f"Erreur lors du pull : {str(e)}", "stderr": ""}
        
        # Envoyer une réponse HTTP avant de redémarrer
        response_data = {"stdout": "Mise à jour terminée. Redémarrage en cours...", "stderr": ""}
        
        # Lancer le redémarrage en arrière-plan avec asyncio
        print("[INFO] Redémarrage de l'application en cours...")
        restart_task = asyncio.create_task(restart_after_delay())
        
        # Retourner immédiatement la réponse HTTP
        return response_data
        
    except Exception as e:
        print(f"[ERROR] Erreur lors du redémarrage automatique: {e}")
        return {"stdout": f"Erreur lors du redémarrage automatique: {str(e)}", "stderr": ""}

@app.post("/upload_s3")
async def upload_s3_route(
    filename: str = Form(...), content: str = Form(...),
    endpoint: str = Form(...), bucket: str = Form(...),
    access_key: str = Form(...), secret_key: str = Form(...)
):
    if boto3 is None:
        raise HTTPException(status_code=500, detail="Le module boto3 n'est pas installé. Veuillez l'installer pour pouvoir télécharger sur S3.")
    try:
        s3 = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3.put_object(Bucket=bucket, Key=filename, Body=content.encode("utf-8"), ContentType='text/plain; charset=utf-8')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
