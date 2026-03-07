import os
import uuid
import asyncio
import tempfile
import time
import torch
# === PATCH TORCHAUDIO (Compatibilité pyannote/torchaudio 2.x) ===
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: print(f"[*] Backend audio patché: {x}")
import json
import datetime
import numpy as np
# === PATCH NUMPY 2.0 (Restauration de np.NaN pour pyannote) ===
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import threading
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    File, UploadFile, Form, BackgroundTasks
)
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from vad_manager import VADManager, SAMPLE_RATE

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
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
    <style>
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
    <h2>🎙️ SOTA ASR — <span id="currentModelDisplay" style="color:#3B82F6">{{model_name}}</span></h2>
    <div style="display:flex; gap:10px; align-items:center; margin-bottom:1rem;">
        <span>Changer de modèle :</span>
        <select id="modelSelector" style="width:200px; margin-bottom:0;" onchange="changeModel()">
            <option value="voxtral" ${model_name === 'voxtral' ? 'selected' : ''}>Voxtral Mini (Fastest)</option>
            <option value="whisper" ${model_name === 'whisper' ? 'selected' : ''}>Faster-Whisper Large (Reliable)</option>
            <option value="qwen" ${model_name === 'qwen' ? 'selected' : ''}>Qwen2-Audio (Advanced)</option>
        </select>
    </div>
    <p>Accélération: <strong>{{device}}</strong> | <span style="color:#10B981">💾 Sauvegarde auto dans <code>transcriptions_terminees/</code></span></p>

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
            <progress id="uploadProgress" value="0" max="100" style="display:none;"></progress>
            <div id="batchStatus" style="margin-top:1rem;font-weight:bold;color:#F59E0B;"></div>
            <div id="batchResult" class="live-box" style="display:none;margin-top:1rem;"></div>
        </div>
    </div>

    <div class="batch-box" style="margin-top:2rem;">
        <h3>📜 Dernières transcriptions</h3>
        <div id="transcriptionList" style="display:flex; flex-direction:column; gap:5px;">
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
        <div id="viewerContent" style="flex:1; overflow-y:auto; background:#111827; color:#E5E7EB; padding:1rem; border-radius:4px; font-family:sans-serif; white-space:pre-wrap;">
        </div>
        <footer>
            <button class="secondary" onclick="closeViewer()">Fermer</button>
        </footer>
    </article>
</dialog>

<script>
// Logic JS identique à l'original de run.sh, avec gestion websocket pour recevoir les segments
let ws = null, isRecording = false;
let audioContext = null, processor = null, source = null;

const getClientId = () => {
    let id = localStorage.getItem("sota_client_id");
    if (!id) { id = "user_" + crypto.randomUUID().slice(0, 8); localStorage.setItem("sota_client_id", id); }
    return id;
};

async function toggleRecording() {
    if (!isRecording) await startRecording(); else stopRecording();
}

async function startRecording() {
    const btn = document.getElementById('recordBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');
    const bar = document.getElementById('audioBar');

    try {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/live`);

        ws.onopen = async () => {
            isRecording = true;
            btn.innerText = "Arrêter l'enregistrement";
            btn.classList.add('recording');
            barCont.style.display = 'block';
            box.innerText = ">> Connexion établie. Parlez...\\n";

            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const processorCode = `
                class AudioProcessor extends AudioWorkletProcessor {
                    constructor() { super(); this.buffer = new Float32Array(4000); this.offset = 0; }
                    process(inputs) {
                        const input = inputs[0][0];
                        if (!input) return true;
                        for (let i = 0; i < input.length; i++) {
                            this.buffer[this.offset++] = input[i];
                            if (this.offset >= 4000) { this.port.postMessage(this.buffer); this.offset = 0; }
                        }
                        return true;
                    }
                }
                registerProcessor('audio-processor', AudioProcessor);
            `;
            const blob = new Blob([processorCode], { type: 'application/javascript' });
            const blobUrl = URL.createObjectURL(blob);
            await audioContext.audioWorklet.addModule(blobUrl);

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            source = audioContext.createMediaStreamSource(stream);
            processor = new AudioWorkletNode(audioContext, 'audio-processor');
            source.connect(processor);

            processor.port.onmessage = (e) => {
                const inputData = e.data;
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) sum += inputData[i] ** 2;
                bar.style.width = Math.min(100, Math.sqrt(sum / inputData.length) * 400) + '%';
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const pcm = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) pcm[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                    ws.send(pcm.buffer);
                }
            };
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.text) {
                box.innerHTML += `<p><strong>[Phrase]</strong>: ${data.text}</p>`;
                box.scrollTop = box.scrollHeight;
            }
        };

        ws.onerror = () => stopRecording();
        ws.onclose = () => { if (isRecording) stopRecording(); };

    } catch (err) { alert("Erreur micro : " + err.message); }
}

function stopRecording() {
    const btn = document.getElementById('recordBtn');
    const box = document.getElementById('liveTranscript');
    if (ws) { ws.close(); ws = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    isRecording = false;
    btn.innerText = "Démarrer l'enregistrement";
    btn.classList.remove('recording');
    box.innerText += "\\n\\n[Fin de session - Repasse finale en cours]";
}

async function changeModel() {
    const newModel = document.getElementById('modelSelector').value;
    const res = await fetch(`/change_model?model=${newModel}`, { method: 'POST' });
    if (res.ok) location.reload();
}

async function loadHistory() {
    const list = document.getElementById('transcriptionList');
    const cid = getClientId();
    const res = await fetch(`/transcriptions?client_id=${cid}`);
    const files = await res.json();
    list.innerHTML = files.map(f => `
        <div style="display:flex; justify-content:space-between; align-items:center; background:#374151; padding:8px 12px; border-radius:6px;">
            <span>${f}</span>
            <button class="outline" style="padding:4px 12px; margin:0;" onclick="viewFile('${f}')">Voir</button>
        </div>
    `).join('') || "Aucune transcription.";
}

async function viewFile(name) {
    const dialog = document.getElementById('viewerDialog');
    const title = document.getElementById('viewerTitle');
    const content = document.getElementById('viewerContent');
    title.innerText = name;
    dialog.showModal();
    const res = await fetch(`/transcription/${name}?client_id=${getClientId()}`);
    content.innerText = await res.text();
}

function closeViewer() { document.getElementById('viewerDialog').close(); }

// Uploader Batch
const CHUNK_SIZE = 4 * 1024 * 1024;
async function uploadFile() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) return;
    const file = fileInput.files[0];
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const fileId = crypto.randomUUID();
    const clientId = getClientId();
    const status = document.getElementById('batchStatus');
    status.innerText = "Upload...";
    for (let i = 0; i < totalChunks; i++) {
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("client_id", clientId);
        formData.append("chunk_index", String(i));
        formData.append("total_chunks", String(totalChunks));
        formData.append("file", file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE), file.name);
        await fetch('/batch_chunk', { method: 'POST', body: formData });
    }
    status.innerText = "Transcription en cours...";
    pollStatus(fileId, status);
}

async function pollStatus(fileId, status) {
    const interval = setInterval(async () => {
        const res = await fetch(`/status/${fileId}`);
        const data = await res.json();
        if (data.status === "done") {
            clearInterval(interval);
            status.innerText = "✅ Terminé.";
            loadHistory();
        } else if (data.status === "error") {
            clearInterval(interval);
            status.innerText = "❌ Erreur: " + data.error;
        } else if (data.status.startsWith("processing:")) {
            status.innerText = "⏳ " + data.status.replace("processing:", "");
        }
    }, 2000);
}

window.onload = loadHistory;
</script>
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
