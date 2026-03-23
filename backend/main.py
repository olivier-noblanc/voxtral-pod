import os
import uuid
import asyncio
import datetime
import torch
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from backend.config import setup_gpu, setup_warnings, TRANSCRIPTIONS_DIR, TEMP_DIR
from backend.html_ui import HTML_UI
from backend.core.engine import SotaASR
from backend.core.live import LiveSession

app = FastAPI(title="SOTA ASR Server", version="4.0.0")
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
def home():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"    
    no_gpu = "true" if device == "cpu" else "false"
    return HTMLResponse(
        content=HTML_UI
            .replace("{{model_name}}", model_name)
            .replace("{{device}}", device)
            .replace("{{no_gpu}}", no_gpu)
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
            html_parts.append(f"""
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-time">{time_str}</span>
                        <span class="segment-speaker">{speaker}</span>
                    </div>
                    <div class="segment-text">{content}</div>
                </div>
            """)
        elif line.strip():
            # Si la ligne ne correspond pas au format, on l'affiche simplement
            html_parts.append(f"<p>{line}</p>")
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
                    <div id="transcript">
                        {format_transcription(content)}
                    </div>
                </div>
            </div>
        </body>
        </html>
    """)

@app.get("/status/{file_id}")
async def status_route(file_id: str):
    return jobs_db.get(file_id, {"status": "not_found"})

@app.post("/upload_s3")
async def upload_s3_route(
    filename: str = Form(...), content: str = Form(...),
    endpoint: str = Form(...), bucket: str = Form(...),
    access_key: str = Form(...), secret_key: str = Form(...)
):
    try:
        import boto3
        s3 = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        s3.put_object(Bucket=bucket, Key=filename, Body=content.encode("utf-8"), ContentType='text/plain; charset=utf-8')
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
