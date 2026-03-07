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

app = FastAPI(title="SOTA ASR Server", version="2.5.0")

model_name = os.getenv("ASR_MODEL", "voxtral").lower()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

jobs_db: dict = {}

# --- HTML UI (Simplified Reference) ---
HTML_UI = """
<!-- Le contenu HTML original de run.sh doit être placé ici. 
     Pour la modularité, il est recommandé de le mettre dans un dossier /static ou /templates -->
<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>SOTA ASR 2026 - Control Panel</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
</head>
<body>
    <main class="container">
        <h2>🎙️ SOTA ASR — <span id="currentModelDisplay" style="color:#3B82F6">{{model_name}}</span></h2>
        <p>Statut: <strong>Architecture Modulaire (VAD Activated)</strong></p>
        <!-- [Le reste du JS de navigation reste identique] -->
    </main>
</body>
</html>
"""

class SotaASR:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.vad = VADManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Chargement du modèle '{model_id}' sur {self.device}…")

        if model_id == "voxtral":
            from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
            self.processor = AutoProcessor.from_pretrained(repo_id)
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(repo_id, quantization_config=quant_config, device_map="auto", attn_implementation="sdpa")
        elif model_id == "whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3", device=self.device, compute_type="float16")
        
        # Diarization setup
        self.diarization = None
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from pyannote.audio import Pipeline
            try:
                self.diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                self.diarization.to(torch.device(self.device))
                print("[*] Diarisation Pyannote 3.1 activée.")
            except Exception as e:
                print(f"[!] Erreur Diarisation : {e}")

        # Live State
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.processing_lock = asyncio.Lock()
        self.full_session_audio = []

    async def transcribe_live_chunk(self, audio_bytes: bytes) -> dict:
        """
        Gère le flux live avec segmentation par VAD.
        Contrairement à avant, on attend une phrase complète.
        """
        self.audio_buffer += audio_bytes
        self.full_session_audio.append(audio_bytes)

        # On check tous les 500ms d'audio accumulé (8000 samples @ 16kHz)
        MIN_CHUNK_FOR_VAD = 8000 * 2 # 16000 bytes
        if len(self.audio_buffer) < MIN_CHUNK_FOR_VAD:
            return {"type": "buffering"}

        chunk_to_test = bytes(self.audio_buffer)
        
        if self.vad.is_speech(chunk_to_test):
            # Parole détectée, on continue d'accumuler
            return {"type": "speech_detected", "text": "..."}
        else:
            # Silence détecté : c'est peut-être la fin d'une phrase
            if len(self.audio_buffer) > 16000: # Au moins 1s de contenu
                async with self.processing_lock:
                    text = await self._run_inference(self.audio_buffer)
                    self.audio_buffer = bytearray() # On vide après transcription
                    return {"type": "sentence", "text": text}
            return {"type": "silence"}

    async def _run_inference(self, pcm_data: bytes) -> str:
        """Inférence réelle sur le chunk audio."""
        import numpy as np
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.model_id == "voxtral":
            inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128)
                return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        elif self.model_id == "whisper":
            # Transcription via faster-whisper (nécessite d'écrire dans un fichier temporaire ou conversion)
            # Simplifié ici pour la structure
            return "Whisper live transcription result"
        return ""

    async def finalize_live_session(self, client_id: str):
        # Repasse haute qualité sur self.full_session_audio
        print(f"[*] Finalizing session for {client_id}")
        self.full_session_audio = []

# Global engine
asr_engine = SotaASR(model_name)

@app.get("/")
def home():
    return HTMLResponse(content=HTML_UI.replace("{{model_name}}", model_name))

@app.websocket("/live")
async def live_endpoint(websocket: WebSocket, background_tasks: BackgroundTasks):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            res = await asr_engine.transcribe_live_chunk(data)
            if res.get("type") == "sentence":
                await websocket.send_json(res)
    except WebSocketDisconnect:
        background_tasks.add_task(asr_engine.finalize_live_session, "anonymous")

@app.post("/batch_chunk")
async def batch_chunk(...): # [Logique de chunking identique à run.sh]
    pass

@app.get("/status/{file_id}")
async def status(file_id: str):
    return JSONResponse(content=jobs_db.get(file_id, {"status": "not_found"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
