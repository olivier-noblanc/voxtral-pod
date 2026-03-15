HTML_UI = r"""<!DOCTYPE html>
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
    <article style="width: 95vw; max-width: 1600px; height: 95vh; display: flex; flex-direction: column; margin: 2.5vh auto; padding: 0;">
        <header style="padding: 1rem 1.5rem; margin-bottom: 0;">
            <a href="#close" aria-label="Close" class="close" onclick="closeViewer()"></a>
            <h4 id="viewerTitle" style="margin:0;">Transcription</h4>
        </header>
        <div id="viewerContent" style="flex:1; min-height:300px; overflow-y:auto; background:#111827; color:#E5E7EB; padding:1rem; border-radius:4px; font-family:sans-serif; white-space:pre-wrap; font-size: 0.95rem; border: 1px solid #374151;">
        </div>
        <footer style="padding: 1rem 1.5rem; margin-top: 0;">
            <div id="exportConfig" style="background:#1F2937; padding:1rem; border-radius:8px; margin-bottom:1rem; border:1px solid #374151; font-size: 0.9rem;">
                <h6 style="margin-bottom:0.75rem; color:#F59E0B; display:flex; align-items:center; gap:8px;">
                    <span style="font-size:1.2rem;">⚙️</span> Configuration de l'export
                </h6>
                <div style="display:flex; flex-wrap:wrap; gap:20px; align-items:center; margin-bottom:1rem; padding-bottom:1rem; border-bottom:1px solid #374151;">
                    <label style="margin:0; display:flex; align-items:center; gap:10px; cursor:pointer; font-weight:bold;">
                        <input type="checkbox" id="includeTimestamps" checked onchange="updateExportPreview()" style="margin:0; width:18px; height:18px;">
                        <span>Inclure les timestamps</span>
                    </label>
                </div>
                <div id="speakerRenameContainer" style="display:none;">
                    <p style="margin-bottom:0.75rem; font-weight:bold; color:#9CA3AF;">Renommer les speakers détectés :</p>
                    <div id="speakerRenameList" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; max-height:180px; overflow-y:auto; padding-right:10px;">
                        <!-- Dynamically filled -->
                    </div>
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; gap:10px;">
                    <button id="btnSaveS3" onclick="uploadToS3()" style="margin:0; background-color:#F59E0B; border-color:#F59E0B; color:black; padding:8px 16px;">Sauvegarder S3</button>
                    <button onclick="copyToClipboard()" style="margin:0; background-color:#10B981; border-color:#10B981; padding:8px 16px;">📋 Copier</button>
                </div>
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
            audioStream = await navigator.mediaDevices.getDisplayMedia({
                video: { displaySurface: "browser" },
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });
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
                if (partialSpan) { partialSpan.remove(); partialSpan = null; }

                if (data.final) {
                    const idx = sentenceIdx++;
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
                        spkSpan.style.visibility = "hidden";
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
                for (const [sidx, spk] of Object.entries(data.speakers)) {
                    const labels = box.querySelectorAll(`.speaker-label[data-sidx="${String(sidx)}"]`);
                    labels.forEach(el => {
                        el.textContent = `[${spk}] `;
                        el.style.visibility = "visible"; el.style.fontSize = "inherit";
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
    btnRecord.innerText = "Démarrer Micro"; btnRecord.disabled = false; btnRecord.classList.remove('recording');
    btnSystem.innerText = "Capturer Réunion (Teams/Browser)"; btnSystem.disabled = false; btnSystem.classList.remove('recording');
    barCont.style.display = 'none';
    
    const partials = box.getElementsByClassName('partial-text');
    while(partials.length > 0) partials[0].parentNode.removeChild(partials[0]);
    
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

let currentRawText = "";

async function viewFile(name) {
    const dialog = document.getElementById('viewerDialog');
    const title = document.getElementById('viewerTitle');
    const content = document.getElementById('viewerContent');
    title.innerText = name; content.innerText = "Chargement...";
    dialog.showModal();
    const res = await fetch(`/transcription/${name}?client_id=${getClientId()}`);
    currentRawText = await res.text();
    
    detectSpeakers(currentRawText);
    updateExportPreview();
}

function detectSpeakers(text) {
    const container = document.getElementById('speakerRenameList');
    const wrapper = document.getElementById('speakerRenameContainer');
    container.innerHTML = "";
    
    // Pattern: [SPEAKER_00]
    const speakerRegex = /\[(SPEAKER_\d+)\]/g;
    const speakers = new Set();
    let match;
    while ((match = speakerRegex.exec(text)) !== null) {
        speakers.add(match[1]);
    }
    
    if (speakers.size > 0) {
        wrapper.style.display = "block";
        speakers.forEach(spk => {
            const div = document.createElement('div');
            div.style.display = "grid";
            div.style.gridTemplateColumns = "100px 1fr";
            div.style.alignItems = "center";
            div.style.gap = "8px";
            div.style.background = "#111827";
            div.style.padding = "6px 10px";
            div.style.borderRadius = "6px";
            div.style.border = "1px solid #374151";
            div.innerHTML = `
                <span style="font-size:0.8rem; color:#F59E0B; font-weight:bold; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${spk}">${spk}</span>
                <input type="text" value="${spk}" data-original="${spk}" oninput="updateExportPreview()" 
                       style="margin:0; padding:4px 8px; font-size:0.85rem; height:auto; background:#1F2937; border:1px solid #4B5563; color:white;">
            `;
            container.appendChild(div);
        });
    } else {
        wrapper.style.display = "none";
    }
}

function updateExportPreview() {
    let text = currentRawText;
    const includeTimestamps = document.getElementById('includeTimestamps').checked;
    
    // Apply speaker renaming
    const inputs = document.querySelectorAll('#speakerRenameList input');
    inputs.forEach(input => {
        const original = input.getAttribute('data-original');
        const replacement = input.value.trim() || original;
        // Global replace for [SPEAKER_XX]
        const escapedOriginal = original.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp('\\[' + escapedOriginal + '\\]', 'g');
        text = text.replace(re, `[${replacement}]`);
    });
    
    // Filter timestamps if needed
    if (!includeTimestamps) {
        // Pattern: [0.76s -> 1.40s] 
        text = text.replace(/\[\d+\.?\d*s -> \d+\.?\d*s\]\s*/g, "");
    }
    
    document.getElementById('viewerContent').innerText = text;
}

function closeViewer() {
    document.getElementById('viewerDialog').close();
}

function copyToClipboard() {
    const content = document.getElementById('viewerContent').innerText;
    navigator.clipboard.writeText(content).then(() => {
        const btn = event.target;
        const oldText = btn.innerText;
        btn.innerText = "✓ Copié !";
        setTimeout(() => { btn.innerText = oldText; }, 2000);
    }).catch(err => {
        alert("Erreur de copie : " + err);
    });
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
            btn.innerText = "Annulation..."; btn.disabled = true;
        }
    } else { uploadFile(); }
}

async function uploadFile() {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput.files.length) return alert("Sélectionner un fichier.");
    const uploadBtn = document.getElementById('uploadBtn');
    const file = fileInput.files[0];
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    
    uploadBtn.innerText = "Upload..."; uploadBtn.disabled = true; fileInput.disabled = true;
    const fileId = crypto.randomUUID();
    const clientId = getClientId();
    const status = document.getElementById('batchStatus');
    const progress = document.getElementById('uploadProgress');
    progress.style.display = 'block'; progress.value = 0; status.innerText = "Upload...";
    
    for (let i = 0; i < totalChunks; i++) {
        const formData = new FormData();
        formData.append("file_id", fileId); formData.append("client_id", clientId);
        formData.append("chunk_index", String(i)); formData.append("total_chunks", String(totalChunks));
        formData.append("file", file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE), file.name);
        const res = await fetch('/batch_chunk', { method: 'POST', body: formData });
        if (!res.ok) { 
            status.innerText = `❌ Erreur chunk ${i+1}`; uploadBtn.innerText = "Envoyer & Transcrire";
            uploadBtn.disabled = false; fileInput.disabled = false; return; 
        }
        progress.value = ((i + 1) / totalChunks) * 100;
    }
    
    uploadBtn.innerText = "Annuler"; uploadBtn.disabled = false; uploadBtn.classList.add('secondary');
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
                clearInterval(interval); localStorage.removeItem('active_batch_file_id');
                uploadBtn.innerText = "Envoyer & Transcrire"; uploadBtn.disabled = false;
                uploadBtn.classList.remove('secondary'); fileInput.disabled = false;
            };
            if (data.status === "done") { resetUI(); status.innerText = "✅ Terminé."; loadHistory();
            } else if (data.status === "cancelled") { resetUI(); status.innerText = "⚠️ Annulé.";
            } else if (data.status === "error") { resetUI(); status.innerText = "❌ Erreur: " + data.error;
            } else if (data.status === "not_found") { resetUI(); status.innerText = "⚠️ Expiré.";
            } else if (data.status && data.status.startsWith("processing:")) {
                let text = "⏳ " + data.status.replace("processing:", "");
                if (data.progress > 0) {
                    text += ` (${data.progress}%)`;
                    if (data.eta > 0) text += ` - Temps restant: ~${formatETA(data.eta)}`;
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
    const btn = document.getElementById('btnSaveS3');
    const oldText = btn.innerText; btn.innerText = "Envoi..."; btn.disabled = true;
    try {
        const formData = new FormData();
        formData.append("filename", title); formData.append("content", content);
        formData.append("endpoint", document.getElementById('s3Endpoint').value);
        formData.append("bucket", document.getElementById('s3Bucket').value);
        formData.append("access_key", document.getElementById('s3AccessKey').value);
        formData.append("secret_key", document.getElementById('s3SecretKey').value);
        const res = await fetch('/upload_s3', { method: 'POST', body: formData });
        if (res.ok) alert("✓ Sauvegardé sur S3 !");
        else alert("❌ Erreur S3 : " + (await res.json()).detail);
    } catch (e) { alert("❌ Erreur réseau : " + e.message); }
    finally { btn.innerText = oldText; btn.disabled = false; }
}

window.onload = () => {
    loadHistory(); loadS3Config();
    const sel = document.getElementById('modelSelector');
    for (let o of sel.options) { if (o.value === '{{model_name}}') o.selected = true; }
    const activeFileId = localStorage.getItem('active_batch_file_id');
    if (activeFileId) {
        const btn = document.getElementById('uploadBtn');
        btn.innerText = "Annuler"; btn.classList.add('secondary');
        document.getElementById('audioFile').disabled = true;
        const statusBox = document.getElementById('batchStatus');
        statusBox.innerText = "Reprise..."; pollStatus(activeFileId, statusBox);
    }
};
</script>
</body>
</html>
"""
