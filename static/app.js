// ========================= VARIABLES GLOBALES =========================
 

let ws, audioContext, source, processor, isRecording = false;
let audioStream = null;
let captureType = "mic";
const CHUNK_SIZE = 4 * 1024 * 1024;
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

// ========================= UTILITAIRES =========================
const getClientId = () => {
    let id = localStorage.getItem("sota_client_id");
    if (!id) { id = "user_" + crypto.randomUUID().slice(0, 8); localStorage.setItem("sota_client_id", id); }
    return id;
};
function updateVolumeBar(data) {
    let sum = 0; for (let i = 0; i < data.length; i++) sum += data[i] ** 2;
    document.getElementById('audioBar').style.width = Math.min(100, Math.sqrt(sum / data.length) * 400) + '%';
}

// ========================= GESTION MICRO / SYSTÈME =========================
async function toggleMicrophone() {
    if (isRecording) stopRecording();
    else { captureType = "mic"; startRecording(); }
}
async function toggleSystemAudio() {
    if (isRecording) stopRecording();
    else { captureType = "system"; startRecording(); }
}

async function startRecording() {
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');
    try {
        if (captureType === "system") {
            audioStream = await navigator.mediaDevices.getDisplayMedia({ video:true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        } else {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        isRecording = true;
        btnRecord.innerText = (captureType === "mic") ? "⏹️ Arrêter" : "🎤 Micro";
        btnSystem.innerText = (captureType === "system") ? "⏹️ Arrêter" : "🖥️ Système";
        if (captureType === "mic") { btnSystem.disabled = true; btnRecord.classList.add('recording'); }
        else { btnRecord.disabled = true; btnSystem.classList.add('recording'); }
        barCont.style.display = 'block';
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${location.host}/live`;
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';
        audioContext = new AudioContext({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(audioStream);
        await audioContext.audioWorklet.addModule('data:text/javascript;base64,' + btoa(workletCode));
        processor = new AudioWorkletNode(audioContext, 'audio-processor');
        processor.port.onmessage = (e) => {
            if (ws && ws.readyState === 1) {
                const pcm = new Int16Array(e.data.length);
                for (let i = 0; i < e.data.length; i++) pcm[i] = Math.max(-1, Math.min(1, e.data[i])) * 0x7FFF;
                ws.send(pcm.buffer);
            }
            updateVolumeBar(e.data);
        };
        source.connect(processor); processor.connect(audioContext.destination);
        box.innerHTML = "";
        let lastSpeaker = "";
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sentence") {
                const row = document.createElement("div"); row.className = "sentence-row " + (data.final ? "finalized" : "");
                const s = document.createElement("span"); s.className = "speaker-label";
                if (data.speaker !== lastSpeaker) { s.textContent = `[${data.speaker}] `; lastSpeaker = data.speaker; }
                const t = document.createElement("span"); t.className = data.final ? "final-text" : "partial-text"; t.textContent = (data.final ? "" : "... ") + data.text;
                row.append(s, t); box.appendChild(row); box.scrollTop = box.scrollHeight;
            } else if (data.type === "final_done") { loadHistory(); }
        };
    } catch (err) { alert(err.message); stopRecording(); }
}

function stopRecording() {
    // Save live transcription if any
    const transcriptBox = document.getElementById('liveTranscript');
    const transcriptText = transcriptBox.innerText.trim();
    if (transcriptText) {
        const clientId = getClientId();
        fetch(`/save_live_transcription/${clientId}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: new URLSearchParams({content: transcriptText})
        }).then(res => {
            if (!res.ok) console.error('Failed to save live transcription');
            else loadHistory();
        }).catch(err => console.error(err));
    }
    if (ws) { ws.close(); ws = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (audioStream) { audioStream.getTracks().forEach(track => track.stop()); audioStream = null; }
    isRecording = false;
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    btnRecord.innerText = "🎤 Micro";
    btnSystem.innerText = "🖥️ Système";
    btnRecord.classList.remove('recording');
    btnSystem.classList.remove('recording');
    btnRecord.disabled = false;
    btnSystem.disabled = false;
    document.getElementById('audioBarCont').style.display = 'none';
}

// ========================= HISTORIQUE =========================
async function loadHistory() {
    const list = document.getElementById('transcriptionList');
    try {
        const res = await fetch(`/transcriptions?client_id=${getClientId()}`);
        const files = await res.json();
list.innerHTML = files.map(f => {
    // Déterminer si c'est un fichier de transcription batch ou live
    const isBatch = f.startsWith('batch_');
    let audioFilename;
    if (isBatch) {
        // Le nom du fichier audio batch inclut l'ID client : batch_<client_id>_<timestamp>.wav
        const ts = f.replace('batch_', '').replace('.txt', '');
        audioFilename = `batch_${getClientId()}_${ts}.wav`;
    } else {
        // Pour les transcriptions live, le fichier audio inclut l'ID client
        const ts = f.replace('live_', '').replace('.txt', '');
        audioFilename = `live_${getClientId()}_${ts}.wav`;
    }
    const downloadUrl = `/download_audio/${getClientId()}/${audioFilename}`;
    const transcriptUrl = `/download_transcript/${getClientId()}/${f}`;
    return `
        <div class="fr-col-12 fr-col-md-4">
            <div class="fr-card fr-card--sm" style="padding:1rem; border:1px solid #3a3a3a;">
                <h4 class="fr-card__title" style="font-size:0.8rem">${f}</h4>
                <a href="/view/${getClientId()}/${f}" class="fr-btn fr-btn--sm fr-mt-1w">Voir</a>
                <a href="${downloadUrl}" class="fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w" download>Télécharger audio</a>
                <a href="${transcriptUrl}" class="fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w" download>Télécharger texte</a>
            </div>
        </div>
    `;
}).join('') || "Aucune.";
    } catch { list.innerHTML = "Erreur."; }
}

// ========================= CONFIGURATION MODÈLE =========================
async function changeModel() {
    const newModel = document.getElementById('modelSelector').value;
    const res = await fetch(`/change_model?model=${newModel}`, { method: 'POST' });
    if (res.ok) location.reload();
}

// ========================= CONFIGURATION S3 =========================
function saveS3Config() {
    localStorage.setItem('s3_conf', JSON.stringify({
        e: document.getElementById('s3Endpoint').value,
        b: document.getElementById('s3Bucket').value,
        a: document.getElementById('s3AccessKey').value,
        s: document.getElementById('s3SecretKey').value
    }));
}
function loadS3Config() {
    const c = JSON.parse(localStorage.getItem('s3_conf') || '{}');
    document.getElementById('s3Endpoint').value = c.e || "";
    document.getElementById('s3Bucket').value = c.b || "";
    document.getElementById('s3AccessKey').value = c.a || "";
    document.getElementById('s3SecretKey').value = c.s || "";
}

function toggleS3Config() {
    const cfg = document.getElementById('s3Config');
    if (cfg.style.display === 'none') {
        cfg.style.display = 'flex';
    } else {
        cfg.style.display = 'none';
    }
}

// ========================= OPTIONS ALBERT =========================
function saveAlbertConfig() {
    localStorage.setItem('albert_partial', document.getElementById('albertPartial').checked);
}
function loadAlbertConfig() {
    const partial = localStorage.getItem('albert_partial') === 'true';
    document.getElementById('albertPartial').checked = partial;
    if (document.getElementById('modelSelector').value === 'albert') {
        document.getElementById('albertOptions').style.display = 'block';
    }
}

// Gestionnaire global pour le bouton "Modifier les speakers"
function toggleSpeakerEditor() {
    const container = document.getElementById('speakerRenameContainer');
    if (!container) return;
    if (container.style.display === 'none' || container.style.display === '') {
        container.style.display = 'block';
        initSpeakerEditor(); // Initialise l'éditeur lorsqu'on l'ouvre
    } else {
        container.style.display = 'none';
    }
}

// Initialise l'éditeur de speakers (utilisé pour les transcriptions batch et live)
function initSpeakerEditor() {
    const speakerSpans = document.querySelectorAll('.segment-speaker');
    const speakers = new Set();
    speakerSpans.forEach(function(span) {
        const speaker = span.dataset.speaker;
        if (speaker) speakers.add(speaker);
    });
    const listDiv = document.getElementById('speakerRenameList');
    listDiv.innerHTML = '';
    speakers.forEach(function(speaker) {
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
        input.addEventListener('input', function(e) {
            const newName = e.target.value;
            const oldName = e.target.dataset.currentSpeaker;
            document.querySelectorAll('.segment-speaker[data-speaker="' + oldName + '"]').forEach(function(span) {
                span.textContent = newName;
                span.dataset.speaker = newName;
            });
            e.target.dataset.currentSpeaker = newName;
        });
        colDiv.appendChild(label);
        colDiv.appendChild(input);
        listDiv.appendChild(colDiv);
    });
}

// ========================= BATCH UPLOAD =========================
async function handleBatchAction() { await uploadFile(); }
async function uploadFile() {
    const input = document.getElementById('audioFile');
    if (!input.files.length) return;
    const file = input.files[0];
    const total = Math.ceil(file.size / CHUNK_SIZE);
    const id = crypto.randomUUID();
    // Sauvegarder l'ID du job dans localStorage
    localStorage.setItem('pending_job', id);
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('uploadProgressContainer').style.display = 'block';
    for (let i = 0; i < total; i++) {
        const formData = new FormData();
        formData.append("file_id", id);
        formData.append("client_id", getClientId());
        formData.append("chunk_index", i);
        formData.append("total_chunks", total);
        formData.append("file", file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE));
        await fetch('/batch_chunk', { method: 'POST', body: formData });
        document.getElementById('uploadProgressFill').style.width = ((i + 1) / total * 100) + '%';
    }
    pollStatus(id);
}
async function pollStatus(id) {
    const interval = setInterval(async () => {
        const res = await fetch(`/status/${id}`);
        const data = await res.json();
        if (data.status === "done") {
            clearInterval(interval);
            // Nettoyer localStorage quand le job est terminé
            localStorage.removeItem('pending_job');
            document.getElementById('uploadBtn').disabled = false;
            document.getElementById('batchStatus').innerText = "✓ Terminé.";
            loadHistory();
        } else if (data.status === "not_found") {
            clearInterval(interval);
            localStorage.removeItem('pending_job');
            document.getElementById('uploadBtn').disabled = false;
        } else if (data.status.startsWith("processing:")) {
            const pct = data.progress || 0;
            const eta = data.eta ? ` (ETA: ${data.eta}s)` : '';
            document.getElementById('batchStatus').innerText = `⏳ ${data.status.split(":")[1]} — ${pct}%${eta}`;
            document.getElementById('uploadProgressFill').style.width = pct + '%';
            document.getElementById('uploadProgressContainer').style.display = 'block';
        }
    }, 2000);
}

// ========================= INITIALISATION =========================
window.onload = () => {
    loadHistory();
    loadS3Config();
loadAlbertConfig();
// Event listeners (replacing inline attributes)
document.getElementById('recordBtn').addEventListener('click', toggleMicrophone);
document.getElementById('systemBtn').addEventListener('click', toggleSystemAudio);
document.getElementById('uploadBtn').addEventListener('click', handleBatchAction);
document.getElementById('toggleS3').addEventListener('click', toggleS3Config);
document.getElementById('modelSelector').addEventListener('change', changeModel);
document.getElementById('albertPartial').addEventListener('change', saveAlbertConfig);
document.getElementById('s3Endpoint').addEventListener('change', saveS3Config);
document.getElementById('s3Bucket').addEventListener('change', saveS3Config);
document.getElementById('s3AccessKey').addEventListener('change', saveS3Config);
document.getElementById('s3SecretKey').addEventListener('change', saveS3Config);
document.getElementById('toggleSpeakerEditorBtn').addEventListener('click', toggleSpeakerEditor);
    // Ajout du gestionnaire pour le bouton "Modifier les speakers"
    // (déplacé en dehors de window.onload, fonction globale définie plus haut)
    document.getElementById('uploadBtn').disabled = false;

    // Reprendre un job en cours si présent
    const pendingJob = localStorage.getItem('pending_job');
    if (pendingJob) {
        document.getElementById('uploadProgressContainer').style.display = 'block';
        document.getElementById('batchStatus').innerText = '⏳ Reprise du job en cours...';
        document.getElementById('uploadBtn').disabled = true;
        pollStatus(pendingJob);
    }

    const noGpu = '{{no_gpu}}' === 'true';
    const selector = document.getElementById('modelSelector');
    if (noGpu) {
        selector.value = 'albert';
        selector.disabled = true;
        document.getElementById('cpuWarning').style.display = 'block';
        document.getElementById('albertOptions').style.display = 'block';
        const badge = document.getElementById('currentModelDisplay');
        badge.innerText = 'Albert API (mode CPU)';
        badge.className = 'fr-badge fr-badge--error fr-badge--no-icon';
    } else {
        selector.value = '{{model_name}}';
    }
};

// ========================= GIT STATUS CHECK =========================
async function checkGitStatus() {
    try {
        const resp = await fetch('/git_status');
        const data = await resp.json();
        console.log('🔍 git_status:', data);

        if (data.behind > 0) {
            if (confirm(`⚠️ ${data.behind} commit(s) disponible(s). Mettre à jour maintenant ?`)) {
                const updResp = await fetch('/git_update', { method: 'POST' });
                const updData = await updResp.json();
                alert('Mise à jour:\n' + updData.stdout);
                location.reload();
            }
        }
    } catch (e) {
        console.error('Erreur git status:', e);
    }
}
window.addEventListener('load', checkGitStatus);