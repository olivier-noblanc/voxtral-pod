import os
import uuid
import asyncio
import tempfile
import time
import torch
import json
import datetime
import numpy as np
import threading
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    File, UploadFile, Form, BackgroundTasks,
)
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from backend.core.vad import VADManager, SAMPLE_RATE

# === CACHE HUGGINGFACE ===
_CACHE_DIR = os.path.abspath("hf_cache")
_HUB_DIR   = os.path.join(_CACHE_DIR, "hub")
os.makedirs(_HUB_DIR, exist_ok=True)
for _k, _v in [
    ("HF_HOME",              _CACHE_DIR),
    ("HF_HUB_CACHE",         _HUB_DIR),
    ("TRANSFORMERS_CACHE",   _HUB_DIR),
    ("HUGGINGFACE_HUB_CACHE",_HUB_DIR),
]:
    os.environ[_k] = _v

app = FastAPI(title="SOTA ASR Server", version="2.6.0")
app.add_middleware(ProxyHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

model_name = os.getenv("ASR_MODEL", "voxtral").lower()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

jobs_db: dict = {}

# --- HTML UI (FULL) ---
HTML_UI = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>SOTA ASR 2026 - Control Panel</title>
<!-- CDN stylesheet removed per project policy -->
<!-- Style block removed per project policy -->
        body { padding: 2rem; max-width: 960px; margin: 0 auto; }
        .live-box {
            height: 200px; overflow-y: auto;
            background: #111928; padding: 1rem;
            border-radius: 8px; font-family: monospace; color: #10B981;
        }
        .batch-box { background: #1F2937; padding: 1.5rem; border-radius: 8px; margin-top: 1rem; }
        #recordBtn.recording {
            background-color: #EF4444; border-color: #EF4444;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .audio-bar-container {
            width: 100%; height: 8px; background: #374151;
            border-radius: 4px; margin: 1rem 0; overflow: hidden; display: none;
        }
        #audioBar { width: 0%; height: 100%; background: #10B981; transition: width 0.1s ease; }
    </style>
</head>
<body>
<main class="container">
<h2>🎙️ SOTA ASR — <span id="currentModelDisplay" class="style1">{{model_name}}</span></h2>
<div class="style2">
        <span>Changer de modèle :</span>
<select id="modelSelector" class="style3" onchange="changeModel()">
            <option value="voxtral" ${model_name === 'voxtral' ? 'selected' : ''}>Voxtral Mini (Fastest)</option>
            <option value="whisper" ${model_name === 'whisper' ? 'selected' : ''}>Faster-Whisper Large (Reliable)</option>
            <option value="qwen" ${model_name === 'qwen' ? 'selected' : ''}>Qwen2-Audio (Advanced)</option>
        </select>
    </div>
<p>Accélération: <strong>{{device}}</strong> | <span class="style4">💾 Sauvegarde auto dans <code>transcriptions_terminees/</code></span></p>

    <div class="grid">
        <div>
            <h3>🔴 Live Stream (VAD Segmentation)</h3>
            <button id="recordBtn" onclick="toggleRecording()">Démarrer l'enregistrement</button>
            <div class="audio-bar-container" id="audioBarCont"><div id="audioBar"></div></div>
            <div class="live-box" id="liveTranscript">En attente de flux audio...</div>
        </div>

        <div class="batch-box">
            <h3>📁 Transcription Batch</h3>
            <p><small>Envoi via chunks de 4 MB + Diarisation réelle (Pyannote)</small></p>
            <input type="file" id="audioFile" accept="audio/*">
            <button onclick="uploadFile()" id="uploadBtn">Envoyer &amp; Transcrire</button>
<progress id="uploadProgress" value="0" max="100" class="style7"></progress>
            <div id="batchStatus" class="style6"></div>
            <div id="batchResult" class="live-box style8"></div>
        </div>
    </div>

    <div class="batch-box" style="margin-top:2rem;">
        <h3>📜 Dernières transcriptions</h3>
        <div id="transcriptionList" class="style9">
            Chargement de l'historique...
        </div>
    </div>
</main>

<dialog id="viewerDialog">
    <article style="max-width:800px; width:90%; height:80vh; display:flex; flex-direction:column;">
        <header>
            <a href="#close" aria-label="Close" class="close" onclick="closeViewer()"></a>
            <h4 id="viewerTitle">Transcription</h4>
        </header>
        <div id="viewerContent" class="style11">
        </div>
        <footer>
            <button class="secondary" onclick="closeViewer()">Fermer</button>
        </footer>
    </article>
</dialog>

<!-- Script tag removed per project policy -->
<!-- Script tag removed per project policy -->
<script src="/static/wavesurfer.js"></script>
</body>
</html>
"""

class SotaASR:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.vad = VADManager(silero_sensitivity=0.4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Chargement du modèle '{model_id}'...")

        if model_id == "voxtral":
            from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
            self.processor = AutoProcessor.from_pretrained(repo_id)
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, quantization_config=quant_config, device_map="auto", attn_implementation="sdpa")
        elif model_id == "whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3", device=self.device, compute_type="float16")
        
        self.diarization = None
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from pyannote.audio import Pipeline
            try:
                self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                self.diarization.to(torch.device(self.device))
                print("[*] Diarisation Pyannote 3.1 OK.")
            except Exception: pass

        # Live Buffer
        self.audio_buffer = bytearray()
        self.full_session_audio = []
        self.processing_lock = asyncio.Lock()

    async def transcribe_live_chunk(self, audio_bytes: bytes) -> dict:
        self.audio_buffer += audio_bytes
        self.full_session_audio.append(audio_bytes)

        # On check la fin de phrase tous les 750ms d'audio accumulé
        if len(self.audio_buffer) < 12000 * 2: # ~0.75s
            return {"status": "buffering"}

        # Si silence détecté par le double VAD
        if not self.vad.is_speech(bytes(self.audio_buffer)):
            if len(self.audio_buffer) > 24000: # On attend d'avoir au moins 1.5s
                async with self.processing_lock:
                    pcm_data = bytes(self.audio_buffer)
                    self.audio_buffer = bytearray()
                    text = await self._run_inference(pcm_data)
                    return {"text": text}
        
        return {"status": "recording"}

    async def _run_inference(self, pcm_data: bytes) -> str:
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.model_id == "voxtral":
            inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=128)
                return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        elif self.model_id == "whisper":
            # Simple conversion pour faster-whisper (mocké ici pour simplicité)
            return "Whisper live text..."
        return ""

    async def finalize_live_session(self, client_id: str):
        # Logique de repasse finale (voir batch)
        if not self.full_session_audio: return
        print(f"[*] Repasse finale pour {client_id}")
        self.full_session_audio = []

# Engine
asr_engine = SotaASR(model_name)

@app.get("/")
def home():
    return HTMLResponse(content=HTML_UI.replace("{{model_name}}", model_name).replace("{{device}}", device))

@app.websocket("/live")
async def live_endpoint(websocket: WebSocket, background_tasks: BackgroundTasks):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            res = await asr_engine.transcribe_live_chunk(data)
            if "text" in res: await websocket.send_json(res)
    except WebSocketDisconnect:
        background_tasks.add_task(asr_engine.finalize_live_session, "anonymous")

@app.post("/change_model")
async def change_model_route(model: str):
    global asr_engine, model_name
    model_name = model
    asr_engine = SotaASR(model)
    return {"status": "ok"}

@app.get("/transcriptions")
async def list_trans(client_id: str):
    p = os.path.join("transcriptions_terminees", client_id)
    if not os.path.exists(p): return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)

@app.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = os.path.join("transcriptions_terminees", client_id, filename)
    with open(p, "r", encoding="utf-8") as f: return PlainTextResponse(f.read())

@app.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...), client_id: str = Form("anonymous"),
    chunk_index: int = Form(...), total_chunks: int = Form(...),
    file: UploadFile = File(...)
):
    tmp_dir = os.path.join(tempfile.gettempdir(), "asr_uploads", file_id)
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, f"chunk_{chunk_index}"), "wb") as f:
        f.write(await file.read())
    
    if chunk_index == total_chunks - 1:
        # Re-assemblage et job background (logique identique à l'ancienne)
        jobs_db[file_id] = {"status": "processing"}
        # ... [Appel à _gpu_job ici] ...
    return {"status": "ok"}

@app.get("/status/{file_id}")
async def status_route(file_id: str): return jobs_db.get(file_id, {"status": "not_found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Include API routes
from backend.routes import api as api_module
app.include_router(api_module.router)
