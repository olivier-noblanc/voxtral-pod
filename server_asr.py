import os
# === CACHE HUGGINGFACE (Must be set BEFORE other imports) ===
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

import uuid
import asyncio
import tempfile
import time
import torch
# === PATCH TORCHAUDIO (pyannote/torchaudio 2.x compat) ===
import torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: print(f"[*] Audio backend patched: {x}")
import json
import datetime
import numpy as np
# === PATCH NUMPY 2.0 (restore np.NaN for pyannote) ===
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import threading
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    File, UploadFile, Form, BackgroundTasks
)
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from vad_manager import VADManager, SAMPLE_RATE

# =====================================================================
# CONFIG
# =====================================================================
app = FastAPI(title="SOTA ASR Server", version="3.0.0")

# Default model = whisper (same as TranscriptionSuite, most reliable for FR)
model_name = os.getenv("ASR_MODEL", "whisper").lower()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

# Hard-coded language
LANGUAGE = "fr"

jobs_db: dict = {}

# =====================================================================
# FRONTEND (HTML UI)
# =====================================================================
HTML_UI = """<!DOCTYPE html>
<html lang="fr" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <title>SOTA ASR — Live Transcription FR</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
    <style>
        body { padding: 2rem; max-width: 960px; margin: 0 auto; }
        .live-box {
            height: 260px; overflow-y: auto;
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
        .speaker-label { color: #60A5FA; font-weight: bold; }
    </style>
</head>
<body>
<main class="container">
    <h2>🎙️ SOTA ASR — <span id="currentModelDisplay" style="color:#3B82F6">{{model_name}}</span></h2>
    <div style="display:flex; gap:10px; align-items:center; margin-bottom:1rem;">
        <span>Modèle :</span>
        <select id="modelSelector" style="width:220px; margin-bottom:0;" onchange="changeModel()">
            <option value="whisper">Faster-Whisper Large-v3 (Recommended)</option>
            <option value="voxtral">Voxtral Mini 4B (Experimental)</option>
        </select>
    </div>
    <p>GPU: <strong>{{device}}</strong> | Langue: <strong>FR</strong> | <span style="color:#10B981">💾 Auto-save → <code>transcriptions_terminees/</code></span></p>

    <div class="grid">
        <div>
            <h3>🔴 Live Transcription (VAD + Diarisation)</h3>
            <button id="recordBtn" onclick="toggleRecording()">Démarrer l'enregistrement</button>
            <div class="audio-bar-container" id="audioBarCont"><div id="audioBar"></div></div>
            <div class="live-box" id="liveTranscript">En attente de flux audio...</div>
        </div>

        <div class="batch-box">
            <h3>📁 Transcription Batch</h3>
            <p><small>Chunks 4 MB + Diarisation Pyannote</small></p>
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
        const clientId = getClientId();
        ws = new WebSocket(`${protocol}//${location.host}/live?client_id=${clientId}`);

        ws.onopen = async () => {
            isRecording = true;
            btn.innerText = "Arrêter l'enregistrement";
            btn.classList.add('recording');
            barCont.style.display = 'block';
            box.innerHTML = "<em>>> Connexion établie. Parlez...</em><br>";

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
            URL.revokeObjectURL(blobUrl);

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

        let lastSpeaker = "";
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sentence" && data.text) {
                const spk = data.speaker || "";
                if (spk && spk !== lastSpeaker) {
                    lastSpeaker = spk;
                    box.innerHTML += `<br><span class="speaker-label">[${spk}]</span> `;
                }
                box.innerHTML += data.text + ' ';
                box.scrollTop = box.scrollHeight;
            } else if (data.type === "final_done") {
                box.innerHTML += `<br><em>[✅ Repasse finale terminée — fichier sauvegardé]</em>`;
                loadHistory();
            }
        };

        ws.onerror = () => stopRecording();
        ws.onclose = () => { if (isRecording) stopRecording(); };

    } catch (err) { alert("Erreur micro : " + err.message); }
}

function stopRecording() {
    const btn = document.getElementById('recordBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');
    if (ws) { ws.close(); ws = null; }
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    isRecording = false;
    btn.innerText = "Démarrer l'enregistrement";
    btn.classList.remove('recording');
    barCont.style.display = 'none';
    box.innerHTML += "<br><em>[Session arrêtée — repasse finale en cours...]</em>";
}

async function changeModel() {
    const newModel = document.getElementById('modelSelector').value;
    const res = await fetch(`/change_model?model=${newModel}`, { method: 'POST' });
    if (res.ok) location.reload();
}

async function loadHistory() {
    const list = document.getElementById('transcriptionList');
    const cid = getClientId();
    try {
        const res = await fetch(`/transcriptions?client_id=${cid}`);
        const files = await res.json();
        if (!files.length) { list.innerHTML = "<p><small>Aucune transcription.</small></p>"; return; }
        list.innerHTML = files.map(f => `
            <div style="display:flex; justify-content:space-between; align-items:center; background:#374151; padding:8px 12px; border-radius:6px;">
                <span>${f}</span>
                <button class="outline" style="padding:4px 12px; margin:0;" onclick="viewFile('${f}')">Voir</button>
            </div>
        `).join('');
    } catch (e) { list.innerHTML = "Erreur chargement historique."; }
}

async function viewFile(name) {
    const dialog = document.getElementById('viewerDialog');
    const title = document.getElementById('viewerTitle');
    const content = document.getElementById('viewerContent');
    title.innerText = name;
    content.innerText = "Chargement...";
    dialog.showModal();
    const res = await fetch(`/transcription/${name}?client_id=${getClientId()}`);
    content.innerText = await res.text();
}

function closeViewer() { document.getElementById('viewerDialog').close(); }

const CHUNK_SIZE = 4 * 1024 * 1024;
async function uploadFile() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) return alert("Sélectionner un fichier.");
    const file = fileInput.files[0];
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const fileId = crypto.randomUUID();
    const clientId = getClientId();
    const status = document.getElementById('batchStatus');
    const progress = document.getElementById('uploadProgress');
    progress.style.display = 'block'; progress.value = 0;
    status.innerText = "Upload...";
    for (let i = 0; i < totalChunks; i++) {
        const formData = new FormData();
        formData.append("file_id", fileId);
        formData.append("client_id", clientId);
        formData.append("chunk_index", String(i));
        formData.append("total_chunks", String(totalChunks));
        formData.append("file", file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE), file.name);
        const res = await fetch('/batch_chunk', { method: 'POST', body: formData });
        if (!res.ok) { status.innerText = `❌ Erreur chunk ${i+1}`; return; }
        progress.value = ((i + 1) / totalChunks) * 100;
    }
    status.innerText = "Transcription en cours...";
    pollStatus(fileId, status);
}

async function pollStatus(fileId, status) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/status/${fileId}`);
            const data = await res.json();
            if (data.status === "done") {
                clearInterval(interval);
                status.innerText = "✅ Terminé.";
                const box = document.getElementById('batchResult');
                box.style.display = 'block';
                box.innerText = JSON.stringify(data.result, null, 2);
                loadHistory();
            } else if (data.status === "error") {
                clearInterval(interval);
                status.innerText = "❌ Erreur: " + data.error;
            } else if (data.status && data.status.startsWith("processing:")) {
                status.innerText = "⏳ " + data.status.replace("processing:", "");
            }
        } catch (e) {}
    }, 2000);
}

// Set selected model on page load
window.onload = () => {
    loadHistory();
    const sel = document.getElementById('modelSelector');
    for (let o of sel.options) { if (o.value === '{{model_name}}') o.selected = true; }
};
</script>
</body>
</html>
"""

# =====================================================================
# ASR ENGINE
# =====================================================================
class SotaASR:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.vad = VADManager(silero_sensitivity=0.4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = SAMPLE_RATE

        # Pre-record buffer: 1 second of audio before speech onset
        self.pre_recording_seconds = 1.0
        self.pre_recording_size = int(self.sample_rate * 2 * self.pre_recording_seconds)

        # Minimum segment length to avoid noise (0.5s = 16000 samples = 32000 bytes)
        self.min_segment_bytes = int(self.sample_rate * 2 * 0.5)

        # Silence threshold: ~1.5s of silence = end of sentence
        # Frontend sends 4000 samples = 250ms per chunk, so 6 chunks ≈ 1.5s
        self.silence_chunks_threshold = 6

        self.processing_lock = asyncio.Lock()

        print(f"[*] Loading model '{model_id}' (v3.0)...")

        if model_id == "whisper":
            from faster_whisper import WhisperModel
            compute_type = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
            print(f"[*] Faster-Whisper: device={self.device}, compute_type={compute_type}")
            self.model = WhisperModel("large-v3", device=self.device, compute_type=compute_type)

        elif model_id == "voxtral":
            from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
            repo_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
            self.processor = AutoProcessor.from_pretrained(repo_id)
            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa"
            )

        # Diarization (Pyannote)
        self.diarization_pipeline = None
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=hf_token
                )
                self.diarization_pipeline.to(torch.device(self.device))
                print("[*] Pyannote Diarization 3.1 loaded OK.")
            except Exception as e:
                print(f"[!] Diarization load failed: {e}")
        else:
            print("[!] HF_TOKEN missing — diarization disabled.")

    # ------------------------------------------------------------------
    # Whisper inference (synchronous, run in thread)
    # ------------------------------------------------------------------
    def _transcribe_sync(self, pcm_bytes: bytes) -> str:
        """Synchronous transcription. Runs in a thread to avoid blocking the event loop."""
        if not pcm_bytes:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if self.model_id == "whisper":
            segments, _ = self.model.transcribe(
                audio,
                beam_size=5,
                language=LANGUAGE,
                task="transcribe"
            )
            return " ".join(s.text for s in segments).strip()

        elif self.model_id == "voxtral":
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(torch.float16)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                    num_beams=1
                )
                return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

        return ""

    # ------------------------------------------------------------------
    # Diarization (synchronous, run in thread)
    # ------------------------------------------------------------------
    def _diarize_sync(self, audio_float32: np.ndarray) -> list:
        """Run pyannote diarization, returns list of (start, end, speaker)."""
        if self.diarization_pipeline is None:
            return []
        waveform = torch.from_numpy(audio_float32[None, :])
        diarization = self.diarization_pipeline(
            {"waveform": waveform, "sample_rate": self.sample_rate}
        )
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return segments

    # ------------------------------------------------------------------
    # Final pass: high-quality transcription + diarization
    # ------------------------------------------------------------------
    def _run_final_pass_sync(self, combined_audio_bytes: bytes) -> str:
        """Full quality pass: diarize + transcribe segment by segment."""
        audio_np = np.frombuffer(combined_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio_np) / self.sample_rate
        print(f"[*] Final pass: {duration:.1f}s of audio")

        # 1. Diarization
        diar_segments = self._diarize_sync(audio_np)

        if not diar_segments:
            # No diarization: transcribe as single block
            if self.model_id == "whisper":
                segments, _ = self.model.transcribe(
                    audio_np, beam_size=5, language=LANGUAGE, task="transcribe"
                )
                lines = []
                for seg in segments:
                    lines.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text.strip()}")
                return "\n".join(lines)
            else:
                return self._transcribe_sync(combined_audio_bytes)

        # 2. Transcribe each diarization segment
        lines = []
        for start, end, speaker in diar_segments:
            s_idx = int(start * self.sample_rate)
            e_idx = int(end * self.sample_rate)
            segment_audio = audio_np[s_idx:e_idx]
            if len(segment_audio) < self.sample_rate * 0.3:
                continue  # Skip tiny segments

            if self.model_id == "whisper":
                segs, _ = self.model.transcribe(
                    segment_audio, beam_size=5, language=LANGUAGE, task="transcribe"
                )
                text = " ".join(s.text for s in segs).strip()
            else:
                pcm = (segment_audio * 32768.0).astype(np.int16).tobytes()
                text = self._transcribe_sync(pcm)

            if text:
                lines.append(f"[{start:.2f}s -> {end:.2f}s] [{speaker}] {text}")

        return "\n".join(lines)

    async def finalize_live_session(self, client_id: str, full_session_audio: list, websocket: WebSocket = None):
        """Final high-quality pass on the full live session audio."""
        if not full_session_audio:
            return
        print(f"[*] Starting final pass for {client_id}")
        try:
            combined = b"".join(full_session_audio)
            transcript = await asyncio.to_thread(self._run_final_pass_sync, combined)
            print(f"=== FINAL TRANSCRIPT ({client_id}) ===\n{transcript}\n{'='*50}")

            # Save
            out_dir = os.path.join("transcriptions_terminees", client_id)
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(out_dir, f"live_{ts}.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"[*] Saved: {filepath}")

            # Notify client if still connected
            if websocket:
                try:
                    await websocket.send_json({"type": "final_done", "file": f"live_{ts}.txt"})
                except Exception:
                    pass

        except Exception as e:
            print(f"[!] Final pass error: {e}")

    # ------------------------------------------------------------------
    # Batch processing (full)
    # ------------------------------------------------------------------
    def _gpu_job_sync(self, audio_path: str, file_id: str, client_id: str):
        """Complete batch job: decode, diarize, transcribe."""
        import subprocess

        def _update(msg):
            if file_id in jobs_db:
                jobs_db[file_id]["status"] = f"processing:{msg}"

        try:
            _update("Décodage audio (ffmpeg)...")
            # Decode to 16kHz mono float32 via ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-i", audio_path, "-f", "f32le", "-acodec", "pcm_f32le",
                 "-ac", "1", "-ar", "16000", "-"],
                capture_output=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")

            audio_np = np.frombuffer(result.stdout, dtype=np.float32).copy()
            duration = len(audio_np) / self.sample_rate
            print(f"[*] Batch: {duration:.1f}s decoded")

            # Diarization
            diar_segments = []
            if self.diarization_pipeline:
                _update("Diarisation en cours...")
                diar_segments = self._diarize_sync(audio_np)
                print(f"[*] Diarization: {len(diar_segments)} segments")

            # Transcription
            _update("Transcription en cours...")
            if self.model_id == "whisper":
                segments, _ = self.model.transcribe(
                    audio_np, beam_size=5, language=LANGUAGE, task="transcribe"
                )
                transcribed_segments = [
                    {"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segments
                ]
            else:
                pcm = (audio_np * 32768.0).astype(np.int16).tobytes()
                text = self._transcribe_sync(pcm)
                transcribed_segments = [{"start": 0, "end": duration, "text": text}]

            # Merge diarization with transcription
            output_lines = []
            for ts_seg in transcribed_segments:
                speaker = "Speaker"
                seg_mid = (ts_seg["start"] + ts_seg["end"]) / 2
                for ds, de, dspk in diar_segments:
                    if ds <= seg_mid <= de:
                        speaker = dspk
                        break
                output_lines.append(
                    f"[{ts_seg['start']:.2f}s -> {ts_seg['end']:.2f}s] [{speaker}] {ts_seg['text']}"
                )

            full_text = "\n".join(output_lines)

            # Save
            out_dir = os.path.join("transcriptions_terminees", client_id)
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_path = os.path.join(out_dir, f"batch_{ts}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            json_path = os.path.join(out_dir, f"batch_{ts}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "text": full_text,
                    "segments": transcribed_segments,
                    "diarization": [{"start": s, "end": e, "speaker": sp} for s, e, sp in diar_segments],
                    "duration": duration,
                    "model": self.model_id,
                    "language": LANGUAGE
                }, f, indent=2, ensure_ascii=False)

            jobs_db[file_id] = {
                "status": "done",
                "result": {"text": full_text, "file": f"batch_{ts}.txt", "duration": duration}
            }
            print(f"[*] Batch done: {txt_path}")

        except Exception as e:
            print(f"[!] Batch error: {e}")
            jobs_db[file_id] = {"status": "error", "error": str(e)}

        finally:
            # Cleanup temp file
            try:
                os.remove(audio_path)
            except Exception:
                pass


# =====================================================================
# LIVE SESSION (one per WebSocket connection)
# =====================================================================
class LiveSession:
    """Isolated per-connection live transcription session with VAD and diarization."""

    def __init__(self, engine: SotaASR, websocket: WebSocket, client_id: str):
        self.engine = engine
        self.websocket = websocket
        self.client_id = client_id

        self.audio_queue = asyncio.Queue()
        self.pre_speech_buffer = bytearray()
        self.sentence_buffer = bytearray()
        self.full_session_audio = []   # ALL audio (speech + silence) for final pass

        self.is_speaking = False
        self.silence_chunks_count = 0

    async def process_audio_queue(self):
        """Worker: dequeue chunks, run dual VAD, trigger inference on sentence boundaries."""
        print(f"[*] [{self.client_id}] Audio processor started.")
        self.engine.vad.reset_states()

        while True:
            audio_bytes = await self.audio_queue.get()
            if audio_bytes is None:
                break

            # Store ALL audio for the final diarization pass
            self.full_session_audio.append(audio_bytes)

            # Update circular pre-record buffer (1s padding)
            self.pre_speech_buffer.extend(audio_bytes)
            if len(self.pre_speech_buffer) > self.engine.pre_recording_size:
                self.pre_speech_buffer = self.pre_speech_buffer[-self.engine.pre_recording_size:]

            # Dual VAD check
            has_speech = self.engine.vad.is_speech(audio_bytes)

            if has_speech:
                if not self.is_speaking:
                    print(f"[*] [{self.client_id}] Speech onset detected.")
                    self.is_speaking = True
                    # Start sentence with pre-record padding
                    self.sentence_buffer = bytearray(self.pre_speech_buffer)
                else:
                    self.sentence_buffer.extend(audio_bytes)
                self.silence_chunks_count = 0

            else:
                if self.is_speaking:
                    self.sentence_buffer.extend(audio_bytes)
                    self.silence_chunks_count += 1

                    if self.silence_chunks_count >= self.engine.silence_chunks_threshold:
                        print(f"[*] [{self.client_id}] End of sentence ({self.silence_chunks_count} silent chunks). Transcribing...")
                        self.is_speaking = False
                        self.silence_chunks_count = 0

                        pcm_data = bytes(self.sentence_buffer)
                        self.sentence_buffer = bytearray()

                        # Skip micro-segments
                        if len(pcm_data) < self.engine.min_segment_bytes:
                            print(f"[*] [{self.client_id}] Segment too short ({len(pcm_data)}B), skipping.")
                            continue

                        # Inference in separate thread (never block the event loop)
                        async with self.engine.processing_lock:
                            try:
                                text = await asyncio.to_thread(self.engine._transcribe_sync, pcm_data)
                                if text:
                                    # Quick speaker estimation via diarization on this segment
                                    speaker = await self._get_speaker_for_segment(pcm_data)
                                    await self.websocket.send_json({
                                        "type": "sentence",
                                        "text": text,
                                        "speaker": speaker,
                                        "final": True
                                    })
                            except Exception as e:
                                print(f"[!] [{self.client_id}] Inference error: {e}")

    async def _get_speaker_for_segment(self, pcm_data: bytes) -> str:
        """Quick diarization on a single sentence segment."""
        if self.engine.diarization_pipeline is None:
            return ""
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_np) < self.engine.sample_rate * 0.5:
                return ""
            segments = await asyncio.to_thread(self.engine._diarize_sync, audio_np)
            if segments:
                # Return the dominant speaker in this segment
                from collections import Counter
                speaker_durations = Counter()
                for start, end, spk in segments:
                    speaker_durations[spk] += end - start
                return speaker_durations.most_common(1)[0][0]
        except Exception as e:
            print(f"[!] Live diarization error: {e}")
        return ""


# =====================================================================
# GLOBAL ENGINE (loaded once in VRAM)
# =====================================================================
asr_engine = SotaASR(model_name)


# =====================================================================
# ROUTES
# =====================================================================
@app.get("/")
def home():
    return HTMLResponse(content=HTML_UI.replace("{{model_name}}", model_name).replace("{{device}}", device))


@app.websocket("/live")
async def live_endpoint(websocket: WebSocket, background_tasks: BackgroundTasks, client_id: str = "anonymous"):
    await websocket.accept()

    session = LiveSession(asr_engine, websocket, client_id)
    processor_task = asyncio.create_task(session.process_audio_queue())

    try:
        while True:
            data = await websocket.receive_bytes()
            await session.audio_queue.put(data)

    except WebSocketDisconnect:
        print(f"[*] [{client_id}] Client disconnected.")

    finally:
        # Stop processor
        await session.audio_queue.put(None)
        await processor_task

        # Launch final pass in background
        if session.full_session_audio:
            background_tasks.add_task(
                asr_engine.finalize_live_session,
                client_id,
                session.full_session_audio,
                websocket
            )


@app.post("/change_model")
async def change_model_route(model: str):
    global asr_engine, model_name
    old = model_name
    model_name = model
    # Free VRAM before loading new model
    del asr_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    asr_engine = SotaASR(model)
    print(f"[*] Model switched: {old} -> {model}")
    return {"status": "ok"}


@app.get("/transcriptions")
async def list_trans(client_id: str):
    p = os.path.join("transcriptions_terminees", client_id)
    if not os.path.exists(p):
        return []
    return sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)


@app.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = os.path.join("transcriptions_terminees", client_id, filename)
    if not os.path.exists(p):
        return PlainTextResponse("Fichier introuvable.", status_code=404)
    with open(p, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read())


@app.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...), client_id: str = Form("anonymous"),
    chunk_index: int = Form(...), total_chunks: int = Form(...),
    file: UploadFile = File(...)
):
    tmp_dir = os.path.join(tempfile.gettempdir(), "asr_uploads", file_id)
    os.makedirs(tmp_dir, exist_ok=True)
    chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())

    if chunk_index == total_chunks - 1:
        # Reassemble chunks into a single file
        jobs_db[file_id] = {"status": "processing:Réassemblage..."}
        assembled_path = os.path.join(tmp_dir, f"full_{file.filename or 'audio.bin'}")
        with open(assembled_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = os.path.join(tmp_dir, f"chunk_{i:04d}")
                with open(cp, "rb") as in_f:
                    out_f.write(in_f.read())
                os.remove(cp)

        # Launch GPU job in background
        background_tasks.add_task(asr_engine._gpu_job_sync, assembled_path, file_id, client_id)

    return {"status": "ok"}


@app.get("/status/{file_id}")
async def status_route(file_id: str):
    return jobs_db.get(file_id, {"status": "not_found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
