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
import warnings
# Ignore specific math warnings from pyannote on very short segments
warnings.filterwarnings("ignore", message="std\(\): degrees of freedom is <= 0")

# === GPU PERFORMANCE OPTIMIZATION (Ampere+) ===
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import torchaudio
import json
import datetime
import numpy as np
# === PATCH NUMPY 2.0 (restore np.NaN for pyannote) ===
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import threading
import re
from collections import Counter
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    File, UploadFile, Form, BackgroundTasks, HTTPException
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
        .sentence-row {
            margin-bottom: 8px;
            display: block;
            border-left: 3px solid transparent;
            padding-left: 10px;
            transition: border-color 0.3s;
        }
        .sentence-row.finalized {
            border-left-color: #3b82f6;
        }
        .speaker-label {
            color: #60a5fa;
            font-weight: 700;
            font-size: 0.85em;
            margin-right: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .partial-text {
            color: #9ca3af;
            font-style: italic;
        }
        .final-text {
            color: #f9fafb;
        }
        .sidebar { background: #1f2937; padding: 20px; border-radius: 8px; height: fit-content; }
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
            <div class="card-header">
                <h2>Live Transcription</h2>
                <div style="display:flex; gap:10px;">
                    <button id="recordBtn" onclick="toggleMicrophone()" style="margin:0;">Démarrer Micro</button>
                    <button id="systemBtn" onclick="toggleSystemAudio()" class="secondary" style="margin:0;">Capturer Réunion (Teams/Browser)</button>
                </div>
            </div>
            <div class="audio-bar-container" id="audioBarCont"><div id="audioBar"></div></div>
            <div class="live-box" id="liveTranscript">En attente de flux audio...</div>
        </div>

        <div class="batch-box">
            <h3>📁 Transcription Batch</h3>
            <p><small>Chunks 4 MB + Diarisation Pyannote</small></p>
            <input type="file" id="audioFile" accept="audio/*">
            <button onclick="handleBatchAction()" id="uploadBtn">Envoyer &amp; Transcrire</button>
            <progress id="uploadProgress" value="0" max="100" style="display:none;"></progress>
            <div id="batchStatus" style="margin-top:1rem;font-weight:bold;color:#F59E0B;"></div>
            <div id="batchResult" class="live-box" style="display:none;margin-top:1rem;"></div>
        </div>
    </div>

    <details style="margin-top: 2rem; background: #1F2937; padding: 1rem; border-radius: 8px;">
        <summary style="cursor: pointer; font-weight: bold; color: #F59E0B;">⚙️ Options d'Export Avancées (S3)</summary>
        <div style="margin-top: 1rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <label>Endpoint URL
                <input type="text" id="s3Endpoint" placeholder="https://s3.eu-west-1.amazonaws.com" onchange="saveS3Config()">
            </label>
            <label>Bucket Name
                <input type="text" id="s3Bucket" placeholder="mon-bucket" onchange="saveS3Config()">
            </label>
            <label>Access Key
                <input type="password" id="s3AccessKey" placeholder="AKIA..." onchange="saveS3Config()">
            </label>
            <label>Secret Key
                <input type="password" id="s3SecretKey" placeholder="wJalr..." onchange="saveS3Config()">
            </label>
        </div>
        <p><small style="color: #9CA3AF;">Ces informations sont sauvegardées localement dans votre navigateur.</small></p>
    </details>

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
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <button id="btnSaveS3" onclick="uploadToS3()" style="margin:0; background-color:#F59E0B; border-color:#F59E0B; color:black; padding:8px 16px;">Sauvegarder sur S3</button>
                <button class="secondary" onclick="closeViewer()" style="margin:0; padding:8px 16px;">Fermer</button>
            </div>
        </footer>
    </article>
</dialog>

<script>
const CHUNK_SIZE = 4 * 1024 * 1024;

const getClientId = () => {
    let id = localStorage.getItem("sota_client_id");
    if (!id) { id = "user_" + crypto.randomUUID().slice(0, 8); localStorage.setItem("sota_client_id", id); }
    return id;
};

const workletCode = `
    class AudioProcessor extends AudioWorkletProcessor {
        constructor() { super(); this.buffer = new Float32Array(2560); this.offset = 0; }
        process(inputs) {
            const input = inputs[0][0];
            if (!input) return true;
            for (let i = 0; i < input.length; i++) {
                this.buffer[this.offset++] = input[i];
                if (this.offset >= 2560) { this.port.postMessage(this.buffer); this.offset = 0; }
            }
            return true;
        }
    }
    registerProcessor('audio-processor', AudioProcessor);
`;

let ws, audioContext, source, processor, isRecording = false;
let audioStream = null;
let captureType = "mic"; // "mic" or "system"

async function toggleMicrophone() {
    if (isRecording) { stopRecording(); return; }
    captureType = "mic";
    startRecording();
}

async function toggleSystemAudio() {
    if (isRecording) { stopRecording(); return; }
    captureType = "system";
    startRecording();
}

async function startRecording() {
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');

    try {
        if (captureType === "system") {
            // navigator.getDisplayMedia for system audio
            audioStream = await navigator.mediaDevices.getDisplayMedia({
                video: { displaySurface: "browser" }, // Try to target a tab/window
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });
            // Ensure user shared audio!
            if (audioStream.getAudioTracks().length === 0) {
                alert("Erreur : Vous devez cocher 'Partager l'audio du système' dans la fenêtre de sélection !");
                audioStream.getTracks().forEach(t => t.stop());
                return;
            }
        } else {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }

        isRecording = true;
        btnRecord.innerText = (captureType === "mic") ? "Arrêter Micro" : "Démarrer Micro";
        btnSystem.innerText = (captureType === "system") ? "Arrêter Réunion" : "Capturer Réunion (Teams/Browser)";
        if (captureType === "mic") { btnSystem.disabled = true; btnRecord.classList.add('recording'); }
        else { btnRecord.disabled = true; btnSystem.classList.add('recording'); }

        barCont.style.display = 'block';
        box.innerHTML = "<em>[Connexion...]</em>";

        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        const cid = getClientId();
        ws = new WebSocket(`${protocol}//${location.host}/live?client_id=${cid}`);
        ws.binaryType = 'arraybuffer';

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(audioStream);
        
        // Load AudioWorklet
        await audioContext.audioWorklet.addModule('data:text/javascript;base64,' + btoa(workletCode));
        processor = new AudioWorkletNode(audioContext, 'audio-processor');
        
        processor.port.onmessage = (e) => {
            const inputData = e.data;
            if (ws && ws.readyState === WebSocket.OPEN) {
                const pcm = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) pcm[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                ws.send(pcm.buffer);
            }
            updateVolumeBar(inputData);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        box.innerHTML = "";
        let lastSpeaker = "";
        let sentenceIdx = 0;
        let partialSpan = null;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sentence" && data.text) {
                const spk = data.speaker || "Speaker";
                
                if (partialSpan) {
                    partialSpan.remove();
                    partialSpan = null;
                }

                if (data.final) {
                    const idx = sentenceIdx++;
                    
                    // Container for the whole sentence line
                    const row = document.createElement("div");
                    row.className = "sentence-row finalized";
                    row.setAttribute("data-sidx", idx);
                    
                    const spkSpan = document.createElement("span");
                    spkSpan.className = "speaker-label";
                    spkSpan.setAttribute("data-sidx", idx);
                    
                    if (spk !== lastSpeaker) {
                        lastSpeaker = spk;
                        spkSpan.textContent = `[${spk}] `;
                    } else {
                        spkSpan.style.visibility = "hidden"; // Keep space but hide for flow
                        spkSpan.style.fontSize = "0px";
                        spkSpan.textContent = `[${spk}] `;
                    }
                    
                    const txtSpan = document.createElement("span");
                    txtSpan.className = "final-text";
                    txtSpan.setAttribute("data-sidx", idx);
                    txtSpan.textContent = data.text;
                    
                    row.appendChild(spkSpan);
                    row.appendChild(txtSpan);
                    box.appendChild(row);
                } else {
                    // Partial container
                    const row = document.createElement("div");
                    row.className = "sentence-row partial-row";
                    
                    partialSpan = document.createElement("span");
                    partialSpan.className = "partial-text";
                    partialSpan.innerText = `... ${data.text}`;
                    
                    row.appendChild(partialSpan);
                    box.appendChild(row);
                }
                box.scrollTop = box.scrollHeight;

            } else if (data.type === "diarization_update" && data.speakers) {
                // Retroactively update speaker labels by index
                for (const [sidx, spk] of Object.entries(data.speakers)) {
                    const idxStr = String(sidx);
                    const labels = box.querySelectorAll(`.speaker-label[data-sidx="${idxStr}"]`);
                    labels.forEach(el => {
                        el.textContent = `[${spk}] `;
                        el.style.visibility = "visible";
                        el.style.fontSize = "inherit";
                    });
                }
            } else if (data.type === "final_done") {
                box.innerHTML += `<br><em>[✅ Repasse finale terminée]</em>`;
                loadHistory();
            }
        };

        ws.onerror = () => stopRecording();
        ws.onclose = () => { if (isRecording) stopRecording(); };

    } catch (err) { alert("Erreur micro : " + err.message); }
}

function stopRecording() {
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');

    if (ws) { ws.close(); ws = null; }
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }

    isRecording = false;
    btnRecord.innerText = "Démarrer Micro";
    btnRecord.disabled = false;
    btnRecord.classList.remove('recording');

    btnSystem.innerText = "Capturer Réunion (Teams/Browser)";
    btnSystem.disabled = false;
    btnSystem.classList.remove('recording');

    barCont.style.display = 'none';
    
    // Clear any lingering partial span
    const partials = box.getElementsByClassName('partial-text');
    while(partials.length > 0){
        partials[0].parentNode.removeChild(partials[0]);
    }
    
    box.innerHTML += "<br><em>[Session arrêtée — repasse finale en cours...]</em>";
}

function updateVolumeBar(inputData) {
    const bar = document.getElementById('audioBar');
    if (!bar) return;
    let sum = 0;
    for (let i = 0; i < inputData.length; i++) sum += inputData[i] ** 2;
    const rms = Math.sqrt(sum / inputData.length);
    bar.style.width = Math.min(100, rms * 400) + '%';
}

function stopRecording() {
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');

    if (ws) { ws.close(); ws = null; }
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }

    isRecording = false;
    btnRecord.innerText = "Démarrer Micro";
    btnRecord.disabled = false;
    btnRecord.classList.remove('recording');

    btnSystem.innerText = "Capturer Réunion (Teams/Browser)";
    btnSystem.disabled = false;
    btnSystem.classList.remove('recording');

    barCont.style.display = 'none';
    
    // Clear any lingering partial span
    const partials = box.getElementsByClassName('partial-text');
    while(partials.length > 0){
        partials[0].parentNode.removeChild(partials[0]);
    }
    
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

function formatETA(totalSeconds) {
    if (totalSeconds < 60) return `${totalSeconds}s`;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    let parts = [];
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    if (seconds > 0 && hours === 0) parts.push(`${seconds}s`);
    return parts.join(' ');
}

async function handleBatchAction() {
    const btn = document.getElementById('uploadBtn');
    const activeFileId = localStorage.getItem('active_batch_file_id');
    
    if (btn.innerText === "Annuler" && activeFileId) {
        if(confirm("Annuler la transcription en cours ?")) {
            await fetch(`/cancel/${activeFileId}`, { method: 'POST' });
            btn.innerText = "Annulation...";
            btn.disabled = true;
        }
    } else {
        uploadFile();
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) return alert("Sélectionner un fichier.");
    const uploadBtn = document.getElementById('uploadBtn');
    const file = fileInput.files[0];
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    
    uploadBtn.innerText = "Upload...";
    uploadBtn.disabled = true;
    fileInput.disabled = true;
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
        if (!res.ok) { 
            status.innerText = `❌ Erreur chunk ${i+1}`; 
            uploadBtn.innerText = "Envoyer & Transcrire";
            uploadBtn.disabled = false;
            fileInput.disabled = false;
            return; 
        }
        progress.value = ((i + 1) / totalChunks) * 100;
    }
    
    uploadBtn.innerText = "Annuler";
    uploadBtn.disabled = false;
    uploadBtn.classList.add('secondary'); // Optional: make it look like a cancel button
    
    status.innerText = "Transcription en cours...";
    localStorage.setItem('active_batch_file_id', fileId);
    pollStatus(fileId, status);
}

async function pollStatus(fileId, status) {
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('audioFile');
    
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/status/${fileId}`);
            const data = await res.json();
            
            const resetUI = () => {
                clearInterval(interval);
                localStorage.removeItem('active_batch_file_id');
                uploadBtn.innerText = "Envoyer & Transcrire";
                uploadBtn.disabled = false;
                uploadBtn.classList.remove('secondary');
                fileInput.disabled = false;
            };

            if (data.status === "done") {
                resetUI();
                status.innerText = "✅ Terminé.";
                const box = document.getElementById('batchResult');
                box.style.display = 'block';
                box.innerText = JSON.stringify(data.result, null, 2);
                loadHistory();
            } else if (data.status === "cancelled") {
                resetUI();
                status.innerText = "⚠️ Annulé par l'utilisateur.";
            } else if (data.status === "error") {
                resetUI();
                status.innerText = "❌ Erreur: " + data.error;
            } else if (data.status === "not_found") {
                resetUI();
                status.innerText = "⚠️ Le fichier a expiré ou le serveur a été redémarré.";
            } else if (data.status && data.status.startsWith("processing:")) {
                let text = "⏳ " + data.status.replace("processing:", "");
                if (data.progress > 0) {
                    text += ` (${data.progress}%)`;
                    if (data.eta > 0) {
                        text += ` - Temps restant: ~${formatETA(data.eta)}`;
                    }
                }
                status.innerText = text;
            }
        } catch (e) {}
    }, 2000);
}

function saveS3Config() {
    localStorage.setItem('s3Endpoint', document.getElementById('s3Endpoint').value);
    localStorage.setItem('s3Bucket', document.getElementById('s3Bucket').value);
    localStorage.setItem('s3AccessKey', document.getElementById('s3AccessKey').value);
    localStorage.setItem('s3SecretKey', document.getElementById('s3SecretKey').value);
}

function loadS3Config() {
    if(localStorage.getItem('s3Endpoint')) document.getElementById('s3Endpoint').value = localStorage.getItem('s3Endpoint');
    if(localStorage.getItem('s3Bucket')) document.getElementById('s3Bucket').value = localStorage.getItem('s3Bucket');
    if(localStorage.getItem('s3AccessKey')) document.getElementById('s3AccessKey').value = localStorage.getItem('s3AccessKey');
    if(localStorage.getItem('s3SecretKey')) document.getElementById('s3SecretKey').value = localStorage.getItem('s3SecretKey');
}

async function uploadToS3() {
    const title = document.getElementById('viewerTitle').innerText;
    const content = document.getElementById('viewerContent').innerText;
    const s3Endpoint = document.getElementById('s3Endpoint').value;
    const s3Bucket = document.getElementById('s3Bucket').value;
    const s3AccessKey = document.getElementById('s3AccessKey').value;
    const s3SecretKey = document.getElementById('s3SecretKey').value;

    if (!s3Bucket || !s3AccessKey || !s3SecretKey) {
        alert("Veuillez configurer les accès S3 dans les 'Options d'Export Avancées' d'abord.");
        return;
    }

    const btn = document.getElementById('btnSaveS3');
    const oldText = btn.innerText;
    btn.innerText = "Envoi...";
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append("filename", title);
        formData.append("content", content);
        formData.append("endpoint", s3Endpoint);
        formData.append("bucket", s3Bucket);
        formData.append("access_key", s3AccessKey);
        formData.append("secret_key", s3SecretKey);

        const res = await fetch('/upload_s3', { method: 'POST', body: formData });
        if (res.ok) {
            alert("✓ Fichier sauvegardé sur S3 !");
        } else {
            const data = await res.json();
            alert("❌ Erreur lors de l'envoi S3 : " + (data.detail || res.statusText));
        }
    } catch (e) {
        alert("❌ Erreur réseau : " + e.message);
    } finally {
        btn.innerText = oldText;
        btn.disabled = false;
    }
}

// Set selected model on page load
window.onload = () => {
    loadHistory();
    loadS3Config();
    const sel = document.getElementById('modelSelector');
    for (let o of sel.options) { if (o.value === '{{model_name}}') o.selected = true; }
    
    // Resume active batch polling if any
    const activeFileId = localStorage.getItem('active_batch_file_id');
    if (activeFileId) {
        const btn = document.getElementById('uploadBtn');
        btn.innerText = "Annuler";
        btn.classList.add('secondary');
        document.getElementById('audioFile').disabled = true;
        
        const statusBox = document.getElementById('batchStatus');
        statusBox.innerText = "Reprise de la transcription...";
        pollStatus(activeFileId, statusBox);
    }
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
        # Frontend sends 2560 samples = 160ms per chunk, so 9 chunks ≈ 1.44s
        self.silence_chunks_threshold = 9

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
        self.diarization_error = None
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
                self.diarization_error = f"Échec du chargement de Pyannote : {e}"
                print(f"[!] {self.diarization_error}")
        else:
            self.diarization_error = "HF_TOKEN manquant dans les variables d'environnement."
            print(f"[!] Diarisation désactivée : {self.diarization_error}")

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
    def _diarize_sync(self, audio_float32: np.ndarray, hook=None) -> list:
        """Run pyannote diarization, returns list of (start, end, speaker)."""
        if self.diarization_pipeline is None:
            return []
        waveform = torch.from_numpy(audio_float32[None, :])
        diarization = self.diarization_pipeline(
            {"waveform": waveform, "sample_rate": self.sample_rate},
            hook=hook
        )
        # Handle Pyannote 4.x DiarizeOutput vs legacy Annotation
        annotation = diarization
        if hasattr(diarization, "annotation"):
            annotation = diarization.annotation

        segments = []
        # pyannote.audio 4.x standard iteration
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return segments

    # ------------------------------------------------------------------
    # Speaker name detection (heuristic on French introductions)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_speaker_names(diar_segments: list, transcribed_lines: list) -> dict:
        """
        Scan transcribed lines for French introduction patterns and map
        anonymous speaker labels (SPEAKER_00) to real names.

        Returns: dict mapping e.g. {"SPEAKER_00": "Pierre", "SPEAKER_01": "Marie"}
        """
        # Patterns that capture a name after a French introduction phrase
        patterns = [
            # "Bonjour, je suis Pierre" / "je m'appelle Marie"
            r"(?:je\s+(?:suis|m['’]appelle))\s+([A-Z\u00C0-\u00FF][a-z\u00E0-\u00FF]+)",
            # "Moi c'est Jean" / "moi, c'est Sophie"
            r"(?:moi[,]?\s+c['’]est)\s+([A-Z\u00C0-\u00FF][a-z\u00E0-\u00FF]+)",
            # "Pierre à l'appareil" / "Ici Marie"
            r"(?:ici)\s+([A-Z\u00C0-\u00FF][a-z\u00E0-\u00FF]+)",
            # "Mon nom est Pierre"
            r"(?:mon\s+nom\s+est)\s+([A-Z\u00C0-\u00FF][a-z\u00E0-\u00FF]+)",
            # "C'est Pierre" (at start of sentence)
            r"^(?:c['’]est)\s+([A-Z\u00C0-\u00FF][a-z\u00E0-\u00FF]+)",
        ]

        name_map = {}
        # Only scan the first few lines per speaker (introductions happen early)
        speaker_line_count = Counter()
        MAX_LINES_TO_SCAN = 5

        for line_info in transcribed_lines:
            speaker = line_info.get("speaker", "")
            text = line_info.get("text", "")
            if not speaker or not text:
                continue
            if speaker in name_map:
                continue  # Already identified

            speaker_line_count[speaker] += 1
            if speaker_line_count[speaker] > MAX_LINES_TO_SCAN:
                continue

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Basic sanity: name should be 2-20 chars, not a common word
                    common_words = {"bonjour", "merci", "alors", "donc", "voilà",
                                    "bien", "tout", "très", "comment", "encore"}
                    if 2 <= len(candidate) <= 20 and candidate.lower() not in common_words:
                        name_map[speaker] = candidate
                        print(f"[*] Speaker identified: {speaker} -> {candidate}")
                        break

        return name_map

    @staticmethod
    def _apply_name_map(text: str, name_map: dict) -> str:
        """Replace anonymous speaker labels with detected names."""
        for anon, real_name in name_map.items():
            text = text.replace(f"[{anon}]", f"[{real_name}]")
        return text

    # ------------------------------------------------------------------
    # Final pass: high-quality transcription + diarization + name detect
    # ------------------------------------------------------------------
    def _run_final_pass_sync(self, combined_audio_bytes: bytes) -> str:
        """Full quality pass: diarize + transcribe segment by segment + detect names."""
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
        structured_lines = []  # For name detection
        output_lines = []
        for start, end, speaker in diar_segments:
            s_idx = int(start * self.sample_rate)
            e_idx = int(end * self.sample_rate)
            segment_audio = audio_np[s_idx:e_idx]
            if len(segment_audio) < self.sample_rate * 0.3:
                continue

            if self.model_id == "whisper":
                segs, _ = self.model.transcribe(
                    segment_audio, beam_size=5, language=LANGUAGE, task="transcribe"
                )
                text = " ".join(s.text for s in segs).strip()
            else:
                pcm = (segment_audio * 32768.0).astype(np.int16).tobytes()
                text = self._transcribe_sync(pcm)

            if text:
                structured_lines.append({"speaker": speaker, "text": text})
                output_lines.append(f"[{start:.2f}s -> {end:.2f}s] [{speaker}] {text}")

        # 3. Detect speaker names and apply
        name_map = self._detect_speaker_names(diar_segments, structured_lines)
        result = "\n".join(output_lines)
        if name_map:
            result = self._apply_name_map(result, name_map)
            # Add a legend at the top
            legend = " | ".join(f"{anon} = {name}" for anon, name in name_map.items())
            result = f"[Speakers: {legend}]\n\n{result}"

        return result

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
        import time

        job_start_time = time.time()

        def _update(msg, progress=0):
            if file_id in jobs_db:
                # Store progress and calculate ETA if progress > 0
                eta = 0
                if progress > 0:
                    elapsed = time.time() - job_start_time
                    total_estimated = elapsed / (progress / 100.0)
                    eta = int(total_estimated - elapsed)
                
                jobs_db[file_id].update({
                    "status": f"processing:{msg}",
                    "progress": progress,
                    "eta": eta
                })

        def _is_cancelled():
            return jobs_db.get(file_id, {}).get("status") == "cancelled"

        try:
            if _is_cancelled(): return
            
            if not self.diarization_pipeline:
                raise RuntimeError(f"Diarisation impossible : {self.diarization_error or 'Erreur inconnue'}. Veuillez vérifier vos clés API.")
                
            print(f"[*] Batch [{file_id[:8]}]: Etape 1/4 - Décodage audio...")
            _update("Etape 1/4 : Décodage audio (ffmpeg)...", 2)
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
            if _is_cancelled(): return
            print(f"[*] Batch [{file_id[:8]}]: Etape 2/4 - Diarisation...")
            
            def _diar_hook(step_name, completed, total=None):
                if _is_cancelled(): return
                # Map diarization (Etape 2) to 5-10% range
                if total:
                    sub_pct = int((completed / total) * 5)
                    _update(f"Etape 2/4 : Diarisation ({step_name})...", 5 + sub_pct)
                else:
                    _update(f"Etape 2/4 : Diarisation ({step_name})...", 5)

            diar_segments = self._diarize_sync(audio_np, hook=_diar_hook)
            print(f"[*] Diarization: {len(diar_segments)} segments")

            # Transcription
            if _is_cancelled(): return
            print(f"[*] Batch [{file_id[:8]}]: Etape 3/4 - Transcription...")
            _update("Etape 3/4 : Transcription en cours...", 10)
            if self.model_id == "whisper":
                segments, info = self.model.transcribe(
                    audio_np, beam_size=5, language=LANGUAGE, task="transcribe"
                )
                transcribed_segments = []
                for s in segments:
                    if _is_cancelled(): return
                    transcribed_segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
                    # Whisper progress roughly from 10% to 90%
                    pct = 10 + min(80, int((s.end / info.duration) * 80)) if info.duration > 0 else 10
                    _update("Etape 3/4 : Transcription en cours...", pct)
                    print(f"[*] Transcription batch [{file_id[:8]}] : {pct}% ({s.end:.1f}s / {info.duration:.1f}s)")

            else:
                pcm = (audio_np * 32768.0).astype(np.int16).tobytes()
                text = self._transcribe_sync(pcm)
                transcribed_segments = [{"start": 0, "end": duration, "text": text}]

            # Merge diarization with transcription
            structured_lines = []
            output_lines = []
            for ts_seg in transcribed_segments:
                speaker = "Speaker"
                seg_mid = (ts_seg["start"] + ts_seg["end"]) / 2
                for ds, de, dspk in diar_segments:
                    if ds <= seg_mid <= de:
                        speaker = dspk
                        break
                structured_lines.append({"speaker": speaker, "text": ts_seg["text"]})
                output_lines.append(
                    f"[{ts_seg['start']:.2f}s -> {ts_seg['end']:.2f}s] [{speaker}] {ts_seg['text']}"
                )

            full_text = "\n".join(output_lines)

            # Detect and apply speaker names
            if _is_cancelled(): return
            print(f"[*] Batch [{file_id[:8]}]: Etape 4/4 - Formatage final...")
            _update("Etape 4/4 : Formatage final...", 95)
            name_map = self._detect_speaker_names(diar_segments, structured_lines)
            if name_map:
                full_text = self._apply_name_map(full_text, name_map)
                legend = " | ".join(f"{a} = {n}" for a, n in name_map.items())
                full_text = f"[Speakers: {legend}]\n\n{full_text}"

            # Save
            if _is_cancelled(): return
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

            jobs_db[file_id].update({
                "status": "done",
                "progress": 100,
                "eta": 0,
                "result": {"text": full_text, "file": f"batch_{ts}.txt", "duration": duration}
            })
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

# Diarization batch interval (seconds of audio accumulated before running Pyannote)
DIAR_BATCH_INTERVAL_SEC = 300  # 5 minutes


class LiveSession:
    """Isolated per-connection live transcription with VAD + periodic batch diarization."""

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

        # Sentence index tracking for speaker assignment
        self._sentence_index = 0
        self._sentences = []  # list of {"idx": int, "text": str, "byte_offset": int}
        self._total_bytes_received = 0

        # Periodic diarization state
        self._last_diar_bytes = 0  # bytes offset of last diarization run
        self._speaker_map = {}     # {sentence_idx: speaker_label}
        self._name_map = {}        # {"SPEAKER_00": "Pierre"}
        self._diar_task = None

    async def process_audio_queue(self):
        """Worker: dequeue chunks, run dual VAD, trigger inference on sentence boundaries and partials."""
        print(f"[*] [{self.client_id}] Audio processor started.")
        self.engine.vad.reset_states()
        
        # Interval for sending partial transcripts (e.g. 2 seconds)
        partial_interval_bytes = int(SAMPLE_RATE * 2 * 2.0)
        last_partial_bytes = 0

        while True:
            audio_bytes = await self.audio_queue.get()
            if audio_bytes is None:
                break

            # Store ALL audio for the final diarization pass
            self.full_session_audio.append(audio_bytes)
            self._total_bytes_received += len(audio_bytes)

            # Update circular pre-record buffer (1s padding)
            self.pre_speech_buffer.extend(audio_bytes)
            if len(self.pre_speech_buffer) > self.engine.pre_recording_size:
                self.pre_speech_buffer = self.pre_speech_buffer[-self.engine.pre_recording_size:]

            # Dual VAD check: strict onset, aggressive offset
            if self.is_speaking:
                has_speech = self.engine.vad.check_deactivation(audio_bytes)
            else:
                has_speech = self.engine.vad.is_speech(audio_bytes)

            if has_speech:
                if not self.is_speaking:
                    print(f"[*] [{self.client_id}] Speech onset detected.")
                    self.is_speaking = True
                    self.sentence_buffer = bytearray(self.pre_speech_buffer)
                    last_partial_bytes = len(self.sentence_buffer)
                else:
                    self.sentence_buffer.extend(audio_bytes)
                self.silence_chunks_count = 0
                
                # Check for partial transcription trigger
                if len(self.sentence_buffer) - last_partial_bytes >= partial_interval_bytes:
                    last_partial_bytes = len(self.sentence_buffer)
                    pcm_data = bytes(self.sentence_buffer)
                    if len(pcm_data) >= self.engine.min_segment_bytes:
                        async with self.engine.processing_lock:
                            try:
                                text = await asyncio.to_thread(self.engine._transcribe_sync, pcm_data)
                                if text:
                                    speaker = self._resolve_speaker(self._sentence_index)
                                    await self.websocket.send_json({
                                        "type": "sentence",
                                        "text": f"{text} ...",
                                        "speaker": speaker,
                                        "final": False
                                    })
                            except Exception as e:
                                print(f"[!] [{self.client_id}] Partial Inference error: {e}")

            else:
                if self.is_speaking:
                    self.sentence_buffer.extend(audio_bytes)
                    self.silence_chunks_count += 1

                    if self.silence_chunks_count >= self.engine.silence_chunks_threshold:
                        print(f"[*] [{self.client_id}] End of sentence. Final Transcribing...")
                        self.is_speaking = False
                        self.silence_chunks_count = 0

                        pcm_data = bytes(self.sentence_buffer)
                        self.sentence_buffer = bytearray()
                        last_partial_bytes = 0

                        if len(pcm_data) < self.engine.min_segment_bytes:
                            continue

                        # Final Inference 
                        async with self.engine.processing_lock:
                            try:
                                text = await asyncio.to_thread(self.engine._transcribe_sync, pcm_data)
                                if text:
                                    idx = self._sentence_index
                                    self._sentence_index += 1
                                    self._sentences.append({
                                        "idx": idx,
                                        "text": text,
                                        "byte_offset": self._total_bytes_received
                                    })

                                    # Use speaker from last diarization batch if available
                                    speaker = self._resolve_speaker(idx)
                                    await self.websocket.send_json({
                                        "type": "sentence",
                                        "text": text,
                                        "speaker": speaker,
                                        "final": True
                                    })
                            except Exception as e:
                                print(f"[!] [{self.client_id}] Inference error: {e}")

            # Check if we should trigger a batch diarization (every 5 min of audio)
            audio_since_last_diar = self._total_bytes_received - self._last_diar_bytes
            threshold_bytes = int(DIAR_BATCH_INTERVAL_SEC * SAMPLE_RATE * 2)  # 16-bit = 2 bytes/sample
            if (audio_since_last_diar >= threshold_bytes
                    and self.engine.diarization_pipeline is not None
                    and (self._diar_task is None or self._diar_task.done())):
                self._diar_task = asyncio.create_task(self._run_batch_diarization())

    async def _run_batch_diarization(self):
        """Run Pyannote diarization on ALL accumulated audio so far (background)."""
        try:
            combined = b"".join(self.full_session_audio)
            self._last_diar_bytes = len(combined)
            audio_np = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(audio_np) / SAMPLE_RATE
            print(f"[*] [{self.client_id}] Batch diarization on {duration:.0f}s of audio...")

            diar_segments = await asyncio.to_thread(self.engine._diarize_sync, audio_np)
            if not diar_segments:
                return

            # Map each sentence to its speaker based on byte_offset -> time
            bytes_per_second = SAMPLE_RATE * 2
            for sent in self._sentences:
                sent_time = sent["byte_offset"] / bytes_per_second
                for ds, de, dspk in diar_segments:
                    if ds <= sent_time <= de:
                        self._speaker_map[sent["idx"]] = dspk
                        break

            # Try to detect speaker names from transcribed sentences
            structured = [{"speaker": self._speaker_map.get(s["idx"], ""), "text": s["text"]}
                          for s in self._sentences]
            self._name_map = SotaASR._detect_speaker_names(diar_segments, structured)

            n_speakers = len(set(s[1] for s in [seg for seg in diar_segments]))
            names_found = len(self._name_map)
            print(f"[*] [{self.client_id}] Batch diarization done: {n_speakers} speakers, {names_found} names identified.")

            # Send a diarization update to the frontend
            try:
                await self.websocket.send_json({
                    "type": "diarization_update",
                    "speakers": {k: self._name_map.get(v, v)
                                 for k, v in self._speaker_map.items()},
                    "name_map": self._name_map
                })
            except Exception:
                pass

        except Exception as e:
            print(f"[!] [{self.client_id}] Batch diarization error: {e}")

    def _resolve_speaker(self, sentence_idx: int) -> str:
        """Get the best speaker label for a sentence (from last batch diarization)."""
        raw = self._speaker_map.get(sentence_idx, "")
        if raw and raw in self._name_map:
            return self._name_map[raw]
        return raw


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
        jobs_db[file_id] = {
            "status": "processing:Réassemblage...",
            "progress": 0,
            "eta": 0
        }
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
    return jobs_db.get(file_id, {"status": "not_found", "progress": 0, "eta": 0})

@app.post("/cancel/{file_id}")
async def cancel_job_route(file_id: str):
    if file_id in jobs_db:
        jobs_db[file_id]["status"] = "cancelled"
    return {"status": "ok"}


@app.post("/upload_s3")
async def upload_s3_route(
    filename: str = Form(...), content: str = Form(...),
    endpoint: str = Form(...), bucket: str = Form(...),
    access_key: str = Form(...), secret_key: str = Form(...)
):
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3_client.put_object(
            Bucket=bucket,
            Key=filename,
            Body=content.encode("utf-8"),
            ContentType='text/plain; charset=utf-8'
        )
        return {"status": "ok", "message": "Uploaded successfully to S3"}
    except Exception as e:
        import traceback
        print("[!] S3 Upload Error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
