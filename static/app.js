// ========================= VARIABLES GLOBALES =========================
let ws, audioContext, source, processor, isRecording = false;
let audioStream = null;
let captureType = "mic";
const CHUNK_SIZE = 4 * 1024 * 1024;
const workletCode = "class AudioProcessor extends AudioWorkletProcessor { constructor() { super(); } process(inputs, outputs, parameters) { const input = inputs[0]; if (input && input[0]) { this.port.postMessage(input[0]); } return true; } } registerProcessor('audio-processor', AudioProcessor);"; // AudioWorklet processor

// ========================= UTILITAIRES =========================
function getClientId() {
    let id = localStorage.getItem("sota_client_id");
    if (!id) {
        id = "user_" + crypto.randomUUID().slice(0, 8);
        localStorage.setItem("sota_client_id", id);
    }
    return id;
}

function updateVolumeBar(data) {
    // Safety check: ensure the audio bar element exists
    const bar = document.getElementById('audioBar');
    if (!bar) {
        console.warn('updateVolumeBar: audioBar element not found');
        return;
    }
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] ** 2;
    const level = Math.min(100, Math.sqrt(sum / data.length) * 400);
    bar.style.width = level + '%';
    // Optional debug output (can be removed later)
    console.debug('Audio level:', level.toFixed(2) + '%');
}

// ========================= GESTION MICRO / SYSTÈME =========================
async function toggleMicrophone() {
        console.log('toggleMicrophone clicked, isRecording:', isRecording);
    if (isRecording) stopRecording();
    else { captureType = "mic"; startRecording(); }
}
async function toggleSystemAudio() {
    if (isRecording) stopRecording();
    else { captureType = "system"; startRecording(); }
}

async function startRecording() {
        console.log('startRecording initiated, captureType:', captureType);
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');
    try {
        if (captureType === "system") {
            audioStream = await navigator.mediaDevices.getDisplayMedia({ video:true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        } else {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('Audio stream obtained (mic), tracks:', audioStream.getTracks().length);
        }
        isRecording = true;
        btnRecord.innerText = (captureType === "mic") ? "⏹️ Arrêter" : "🎤 Micro";
        btnSystem.innerText = (captureType === "system") ? "⏹️ Arrêter" : "🖥️ Système";
        if (captureType === "mic") { btnSystem.disabled = true; btnRecord.classList.add('recording'); }
        else { btnRecord.disabled = true; btnSystem.classList.add('recording'); }
        barCont.style.display = 'block';
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        let wsUrl = wsProtocol + '//' + location.host + '/live';
        const model = document.getElementById('modelSelector').value;
        const partial = document.getElementById('albertPartial').checked;
        if (model === 'albert' && partial) {
            wsUrl += `?client_id=${getClientId()}&partial_albert=true`;
        } else {
            wsUrl += `?client_id=${getClientId()}`;
        }
        ws = new WebSocket(wsUrl);
        console.log('WebSocket opened to', wsUrl);
        ws.binaryType = 'arraybuffer';
        // Use default sample rate to avoid mismatch with MediaStream source
        audioContext = new AudioContext({ sampleRate: 16000 });
        console.log('AudioContext created with sampleRate', audioContext.sampleRate);
        source = audioContext.createMediaStreamSource(audioStream);
        await audioContext.audioWorklet.addModule('data:text/javascript;base64,' + btoa(workletCode));
        console.log('AudioWorklet module added');
        processor = new AudioWorkletNode(audioContext, 'audio-processor');
        console.log('AudioWorkletNode (processor) created');
processor.port.onmessage = (e) => {
    console.log('Audio worklet data received, length:', e.data.length);
    if (ws && ws.readyState === 1) {
        const pcm = new Int16Array(e.data.length);
        for (let i = 0; i < e.data.length; i++) {
            pcm[i] = Math.max(-1, Math.min(1, e.data[i])) * 0x7FFF;
        }
        ws.send(pcm.buffer);
        console.log('PCM data sent');
        updateVolumeBar(e.data);
    } else {
        updateVolumeBar(e.data);
    }
};
        source.connect(processor);
        console.log('MediaStreamSource connected to processor');
        processor.connect(audioContext.destination);
        console.log('Processor connected to audio context destination');
        box.innerHTML = "";
        let lastSpeaker = "";
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sentence") {
                const row = document.createElement("div");
                row.className = "sentence-row " + (data.final ? "finalized" : "");
                const s = document.createElement("span");
                s.className = "speaker-label";
                if (data.speaker !== lastSpeaker) { s.textContent = "[" + data.speaker + "] "; lastSpeaker = data.speaker; }
                const t = document.createElement("span");
                t.className = data.final ? "final-text" : "partial-text";
                t.textContent = (data.final ? "" : "... ") + data.text;
                row.append(s, t);
                box.appendChild(row);
                box.scrollTop = box.scrollHeight;
            } else if (data.type === "final_done") {
                loadHistory();
            }
        };
    } catch (err) {
        alert(err.message);
        stopRecording();
    }
}

function stopRecording() {
    // Save live transcription if any
    const transcriptBox = document.getElementById('liveTranscript');
    const transcriptText = transcriptBox.innerText.trim();
    if (transcriptText) {
        const clientId = getClientId();
        fetch('/save_live_transcription/' + clientId, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ content: transcriptText })
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
        const res = await fetch('/transcriptions?client_id=' + getClientId());
        const files = await res.json();
        list.innerHTML = files.map(function(f) {
            var isBatch = f.startsWith('batch_');
            var audioFilename;
            if (isBatch) {
                let ts = f.replace('batch_', '').replace('.txt', '');
                audioFilename = 'batch_' + getClientId() + '_' + ts + '.wav';
            } else {
                let ts = f.replace('live_', '').replace('.txt', '');
                audioFilename = 'live_' + getClientId() + '_' + ts + '.wav';
            }
            var downloadUrl = '/download_audio/' + getClientId() + '/' + audioFilename;
            var transcriptUrl = '/download_transcript/' + getClientId() + '/' + f;
            return '<div class="fr-col-12 fr-col-md-4">' +
                '<div class="fr-card fr-card--sm" style="padding:1rem; border:1px solid #3a3a3a;">' +
                '<h4 class="fr-card__title" style="font-size:0.8rem">' + f + '</h4>' +
                '<a href="/view/' + getClientId() + '/' + f + '" class="fr-btn fr-btn--sm fr-mt-1w">Voir</a>' +
                '<a href="' + downloadUrl + '" class="fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w" download>Télécharger audio</a>' +
                '<a href="' + transcriptUrl + '" class="fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w" download>Télécharger texte</a>' +
                '</div></div>';
        }).join('') || "Aucune.";
    } catch {
        list.innerHTML = "Erreur.";
    }
}

// ========================= CONFIGURATION MODÈLE =========================
async function changeModel() {
    const newModel = document.getElementById('modelSelector').value;
    const res = await fetch('/change_model?model=' + newModel, { method: 'POST' });
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

// ========================= GESTIONNAIRE SPEAKER =========================
function toggleSpeakerEditor() {
    const container = document.getElementById('speakerRenameContainer');
    if (!container) return;
    if (container.style.display === 'none' || container.style.display === '') {
        container.style.display = 'block';
        initSpeakerEditor();
    } else {
        container.style.display = 'none';
    }
}
function initSpeakerEditor() {
    const speakerSpans = document.querySelectorAll('.segment-speaker');
    const speakers = new Set();
    speakerSpans.forEach(span => {
        const speaker = span.dataset.speaker;
        if (speaker) speakers.add(speaker);
    });
    const listDiv = document.getElementById('speakerRenameList');
    listDiv.innerHTML = '';
    speakers.forEach(speaker => {
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
        input.addEventListener('input', e => {
            const newName = e.target.value;
            const oldName = e.target.dataset.currentSpeaker;
            document.querySelectorAll('.segment-speaker[data-speaker="' + oldName + '"]').forEach(span => {
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
        const res = await fetch('/status/' + id);
        const data = await res.json();
        if (data.status === "done") {
            clearInterval(interval);
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
            const eta = data.eta ? " (ETA: " + data.eta + "s)" : '';
                const statusLabel = (data.status && data.status.includes(":")) ? data.status.split(":")[1] : (data.status || "unknown");
            document.getElementById('batchStatus').innerText = "⏳ " + statusLabel + " — " + pct + "%" + eta;
            document.getElementById('uploadBtn').disabled = true;
            document.getElementById('uploadProgressFill').style.width = pct + '%';
            document.getElementById('uploadProgressContainer').style.display = 'block';
        }
    }, 2000);
}

// ========================= GIT STATUS CHECK =========================
async function checkGitStatus() {
    try {
        const resp = await fetch('/git_status');
        const data = await resp.json();
        console.log('🔍 git_status:', data);
        if (data.behind > 0) {
            if (confirm('⚠️ ' + data.behind + ' commit(s) disponible(s). Mettre à jour maintenant ?')) {
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

// ========================= INITIALISATION =========================
window.onload = () => {
    // Initialise les écouteurs d'événements
    const elRecord = document.getElementById('recordBtn');
    if (elRecord) elRecord.addEventListener('click', toggleMicrophone);
    const elSystem = document.getElementById('systemBtn');
    if (elSystem) elSystem.addEventListener('click', toggleSystemAudio);
    const elUpload = document.getElementById('uploadBtn');
    if (elUpload) elUpload.addEventListener('click', handleBatchAction);
    const elToggleS3 = document.getElementById('toggleS3');
    if (elToggleS3) elToggleS3.addEventListener('click', toggleS3Config);
    const elModel = document.getElementById('modelSelector');
    if (elModel) elModel.addEventListener('change', changeModel);
    const elAlbert = document.getElementById('albertPartial');
    if (elAlbert) elAlbert.addEventListener('click', saveAlbertConfig);
    const elS3Endpoint = document.getElementById('s3Endpoint');
    if (elS3Endpoint) elS3Endpoint.addEventListener('change', saveS3Config);
    const elS3Bucket = document.getElementById('s3Bucket');
    if (elS3Bucket) elS3Bucket.addEventListener('change', saveS3Config);
    const elS3Access = document.getElementById('s3AccessKey');
    if (elS3Access) elS3Access.addEventListener('click', saveS3Config);
    const elS3Secret = document.getElementById('s3SecretKey');
    if (elS3Secret) elS3Secret.addEventListener('click', saveS3Config);
    const elToggleSpeaker = document.getElementById('toggleSpeakerEditorBtn');
    if (elToggleSpeaker) elToggleSpeaker.addEventListener('click', toggleSpeakerEditor);

    // Chargement initial des données
    loadHistory();
    loadS3Config();
    loadAlbertConfig();

    // Forcer Albert si aucun GPU détecté
    const deviceBadge = document.querySelector('.fr-badge.fr-badge--info');
    if (deviceBadge && deviceBadge.textContent.trim() === 'cpu') {
        const modelSelect = document.getElementById('modelSelector');
        if (modelSelect) {
            modelSelect.value = 'albert';
            modelSelect.disabled = true; // désactiver la listbox en mode CPU
            // Afficher l'avertissement CPU
            const cpuWarn = document.getElementById('cpuWarning');
            if (cpuWarn) cpuWarn.style.display = 'block';
            // Mettre à jour l'affichage des options Albert
            loadAlbertConfig();
        }
    }

    // Vérifier les mises à jour Git
    checkGitStatus();

    // Reprendre le suivi d'un batch en cours après rafraîchissement
    const pendingJob = localStorage.getItem('pending_job');
    if (pendingJob) {
        const uploadBtn = document.getElementById('uploadBtn');
        const progressContainer = document.getElementById('uploadProgressContainer');
        const statusElem = document.getElementById('batchStatus');

        // Désactiver le bouton et afficher le conteneur de progression
        if (uploadBtn) uploadBtn.disabled = true;
        if (progressContainer) progressContainer.style.display = 'block';
        if (statusElem) statusElem.innerText = "⏳ Reprise du traitement...";

        // Fonction de secours : réactiver le bouton après 30 s si aucune réponse
        const fallbackTimeout = setTimeout(() => {
            if (uploadBtn) uploadBtn.disabled = false;
            if (statusElem) statusElem.innerText = "⚠️ Aucun statut reçu, veuillez réessayer.";
            loadHistory();
        }, 30000);

        // Vérifier immédiatement le statut du job
        fetch('/status/' + pendingJob)
            .then(res => res.json())
            .then(data => {
                clearTimeout(fallbackTimeout);
                if (data.status === "done") {
                    localStorage.removeItem('pending_job');
                    if (uploadBtn) uploadBtn.disabled = false;
                    if (statusElem) statusElem.innerText = "✓ Terminé.";
                    loadHistory();
                } else if (data.status === "not_found") {
                    localStorage.removeItem('pending_job');
                    if (uploadBtn) uploadBtn.disabled = false;
                    if (statusElem) statusElem.innerText = "";
                    loadHistory();
                } else {
                    // Job en cours
                    if (uploadBtn) uploadBtn.disabled = true;
                if (statusElem) {
                    const pct = data.progress || 0;
                    const eta = data.eta ? " (ETA: " + data.eta + "s)" : '';
                    const statusLabel = (data.status && data.status.includes(":")) ? data.status.split(":")[1] : (data.status || "unknown");
                    statusElem.innerText = "⏳ " + statusLabel + " — " + pct + "%" + eta;
                }
                    pollStatus(pendingJob);
                }
            })
            .catch(() => {
                clearTimeout(fallbackTimeout);
                // En cas d'erreur réseau, démarrer le polling
                if (uploadBtn) uploadBtn.disabled = true;
                if (progressContainer) progressContainer.style.display = 'block';
                pollStatus(pendingJob);
            });
    }
};