// ========================= VARIABLES GLOBALES =========================
let ws, audioContext, source, processor, isRecording = false;
let audioStream = null;
let captureType = "mic";
let selectedAudioDeviceId = "default";
const CHUNK_SIZE = 4 * 1024 * 1024;

// Counter to uniquely identify each transcript segment (used for drag‑and‑drop)
let segmentCounter = 0;
let audioDeviceCount;




const workletCode = `class AudioProcessor extends AudioWorkletProcessor {
    constructor() { super(); this.buffer = new Float32Array(2560); this.offset = 0; }
    process(inputs) {
        const input = inputs[0][0];
        if (!input) return true;
        for (let i = 0; i < input.length; i++) {
            this.buffer[this.offset++] = input[i];
            if (this.offset >= 2560) {
                this.port.postMessage(this.buffer);
                this.offset = 0;
            }
        }
        return true;
    }
}
registerProcessor('audio-processor', AudioProcessor);`; // AudioWorklet processor

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

async function checkRateLimitStatus() {
    const alert = document.getElementById('albertRateLimitAlert');
    const timer = document.getElementById('fallbackTimer');
    const quotaBadge = document.getElementById('albertQuotaBadge');
    const modelSelector = document.getElementById('modelSelector');

    if (!alert || !timer || !quotaBadge || !modelSelector) return;

    try {
        const res = await fetch('/rate_limiter_status');
        const data = await res.json();

        // Affichage du quota uniquement si le modèle Albert est sélectionné
        if (modelSelector.value === 'albert') {
            quotaBadge.style.display = 'inline-flex';
            quotaBadge.innerText = `Quota ASR Albert : ${data.quota_asr_usage} / ${data.quota_limit}`;

            // Changer la couleur selon l'usage ASR
            const ratio = data.quota_asr_usage / data.quota_limit;
            if (ratio > 0.9) {
                quotaBadge.className = 'fr-badge fr-badge--error fr-badge--no-icon fr-ml-1w';
            } else if (ratio > 0.7) {
                quotaBadge.className = 'fr-badge fr-badge--warning fr-badge--no-icon fr-ml-1w';
            } else {
                quotaBadge.className = 'fr-badge fr-badge--info fr-badge--no-icon fr-ml-1w';
            }
        } else {
            quotaBadge.style.display = 'none';
        }

        if (data.fallback_active) {
            alert.style.display = 'block';
            timer.innerText = data.fallback_remaining_seconds;
        } else {
            alert.style.display = 'none';
        }
    } catch (e) {
        // Ignorer silencieusement
    }
}

// ========================= GESTION MICRO / SYSTEME =========================
async function toggleMicrophone() {
    console.log('toggleMicrophone clicked, isRecording:', isRecording);
    if (isRecording) stopRecording();
    else { captureType = "mic"; startRecording(); }
}
async function toggleSystemAudio() {
    if (isRecording) stopRecording();
    else { captureType = "system"; startRecording(); }
}
async function toggleMeeting() {
    if (isRecording) stopRecording();
    else { captureType = "meeting"; startRecording(); }
}

async function startRecording() {
    console.log('startRecording initiated, captureType:', captureType);
    // Get current model from selector
    const modelSelector = document.getElementById('modelSelector');
    if (modelSelector && modelSelector.value) {
        // Optionnel : on pourrait forcer le changement de modèle ici si besoin
        console.log('Using model:', modelSelector.value);
    }

    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const btnMeeting = document.getElementById('meetingBtn');
    const box = document.getElementById('liveTranscript');
    const barCont = document.getElementById('audioBarCont');
    try {
        if (captureType === "system") {
            audioStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
        } else if (captureType === "meeting") {
            // Mixed mode: capture both mic and system
            const micConstraints = { audio: selectedAudioDeviceId === "default" ? true : { deviceId: { exact: selectedAudioDeviceId } } };
            const micStream = await navigator.mediaDevices.getUserMedia(micConstraints);

            const sysStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });

            // Combine tracks into a single stream for easier cleanup, 
            // but we'll create separate sources for mixing in AudioContext.
            audioStream = new MediaStream([...micStream.getTracks(), ...sysStream.getTracks()]);
            audioStream._micStream = micStream;
            audioStream._sysStream = sysStream;
        } else {
            const micConstraints = { audio: selectedAudioDeviceId === "default" ? true : { deviceId: { exact: selectedAudioDeviceId } } };
            audioStream = await navigator.mediaDevices.getUserMedia(micConstraints);

            console.log('Audio stream obtained (mic), tracks:', audioStream.getTracks().length);
        }
        isRecording = true;
        btnRecord.innerText = (captureType === "mic") ? "⏹️ Arrêter" : "🎤 Micro";
        btnSystem.innerText = (captureType === "system") ? "⏹️ Arrêter" : "🖥️ Système";
        btnMeeting.innerText = (captureType === "meeting") ? "⏹️ Arrêter" : "👥 Réunion";

        if (captureType === "mic") { btnSystem.disabled = true; btnMeeting.disabled = true; btnRecord.classList.add('recording'); }
        else if (captureType === "system") { btnRecord.disabled = true; btnMeeting.disabled = true; btnSystem.classList.add('recording'); }
        else { btnRecord.disabled = true; btnSystem.disabled = true; btnMeeting.classList.add('recording'); }
        barCont.style.display = 'block';
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        let wsUrl = wsProtocol + '//' + location.host + '/live';
        wsUrl += '?client_id=' + getClientId();
        const partialCheckbox = document.getElementById('albertPartial');
        const usePartial = partialCheckbox ? partialCheckbox.checked : false;
        wsUrl += '&partial_albert=' + usePartial;
        // Add audio device information to WebSocket URL
        if (selectedAudioDeviceId && selectedAudioDeviceId !== "default") {
            wsUrl += `&device_id=${encodeURIComponent(selectedAudioDeviceId)}`;
        }
        window.currentLiveSessionId = "live_final_" + Date.now();
        wsUrl += '&session_id=' + encodeURIComponent(window.currentLiveSessionId);
        ws = new WebSocket(wsUrl);
        console.log('WebSocket opened to', wsUrl);
        ws.binaryType = 'arraybuffer';
        // Use default sample rate to match the MediaStream source
        audioContext = new AudioContext();
        console.log('AudioContext created with sampleRate', audioContext.sampleRate);

        await audioContext.audioWorklet.addModule('data:text/javascript;base64,' + btoa(workletCode));
        console.log('AudioWorklet module added');
        processor = new AudioWorkletNode(audioContext, 'audio-processor');
        console.log('AudioWorkletNode (processor) created');

        if (captureType === "meeting" && audioStream._micStream && audioStream._sysStream) {
            const micSource = audioContext.createMediaStreamSource(audioStream._micStream);
            const sysSource = audioContext.createMediaStreamSource(audioStream._sysStream);
            micSource.connect(processor);
            sysSource.connect(processor);
            console.log('Mic and System streams mixed into processor');
        } else {
            source = audioContext.createMediaStreamSource(audioStream);
            source.connect(processor);
            console.log('Single stream connected to processor');
        }

        processor.port.onmessage = (e) => {
            if (!audioContext) return;
            //console.log('Audio worklet data received, length:', e.data.length);
            // Dynamically compute down-sampling factor based on the AudioContext sample rate
            const targetRate = 16000;
            const factor = Math.max(1, Math.round(audioContext.sampleRate / targetRate));
            const downLength = Math.floor(e.data.length / factor);
            const downsampled = new Float32Array(downLength);
            for (let i = 0; i < downLength; i++) {
                downsampled[i] = e.data[i * factor];
            }
            const pcm = new Int16Array(downsampled.length);
            for (let i = 0; i < downsampled.length; i++) {
                pcm[i] = Math.max(-1, Math.min(1, downsampled[i])) * 0x7FFF;
            }
            if (ws.readyState === 1) {
                ws.send(pcm.buffer);
            }
            //console.log('PCM data sent (downsampled by factor', factor, ')');
            // Update UI with downsampled data for a responsive volume bar
            updateVolumeBar(e.data);
        };
        // source is already connected inside the if/else block above
        processor.connect(audioContext.destination);
        console.log('Processor connected to audio context destination');
        box.innerHTML = "";
        let lastSpeaker = "";
        let partialRow = null; // Reference to the current in-progress partial row
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sentence") {
                if (data.final) {
                    // Look up partial row by sentence_index first (robust against out-of-order messages)
                    const target = (data.sentence_index != null)
                        ? box.querySelector(`[data-sindex="${data.sentence_index}"]`) || partialRow
                        : partialRow;
                    if (target) {
                        const partialSpan = target.querySelector("span.partial-text");
                        if (partialSpan) {
                            partialSpan.className = "final-text";
                            partialSpan.textContent = data.text;
                        }
                        target.classList.add("finalized");
                        target.removeAttribute("data-sindex");
                        if (partialRow === target) partialRow = null;
                    } else {
                        // No partial in flight: create a new final line directly
                        const row = document.createElement("div");
                        row.className = "sentence-row finalized";
                        const s = document.createElement("span");
                        s.className = "speaker-label";
                        if (data.speaker !== lastSpeaker) { s.textContent = "[" + data.speaker + "] "; lastSpeaker = data.speaker; }
                        const t = document.createElement("span");
                        t.className = "final-text";
                        t.textContent = data.text;
                        row.append(s, t);
                        box.appendChild(row);
                    }
                } else {
                    // Partial segment: update existing partialRow or create one
                    // Partial: find existing row by sentence_index or fall back to partialRow
                    const existingPartial = (data.sentence_index != null)
                        ? box.querySelector(`[data-sindex="${data.sentence_index}"]`) || partialRow
                        : partialRow;
                    if (existingPartial) {
                        // Reuse the same row — just update the text
                        const partialSpan = existingPartial.querySelector("span.partial-text");
                        if (partialSpan) {
                            partialSpan.textContent = data.text;
                            const dots = document.createElement("span");
                            dots.className = "dots-animated";
                            dots.textContent = "...";
                            partialSpan.appendChild(dots);
                        }
                    } else {
                        const row = document.createElement("div");
                        row.className = "sentence-row";
                        row.draggable = true;
                        row.dataset.idx = segmentCounter++;
                        if (data.sentence_index != null) row.dataset.sindex = data.sentence_index;
                        const s = document.createElement("span");
                        s.className = "speaker-label";
                        if (data.speaker !== lastSpeaker) { s.textContent = "[" + data.speaker + "] "; lastSpeaker = data.speaker; }
                        const t = document.createElement("span");
                        t.className = "partial-text";
                        t.textContent = data.text;
                        const dots = document.createElement("span");
                        dots.className = "dots-animated";
                        dots.textContent = "...";
                        t.appendChild(dots);
                        row.append(s, t);
                        // Drag‑and‑drop handlers (same as for finalized rows)
                        row.addEventListener("dragstart", e => {
                            e.dataTransfer.setData("text/plain", e.currentTarget.dataset.idx);
                        });
                        row.addEventListener("dragover", e => e.preventDefault());
                        row.addEventListener("drop", async e => {
                            e.preventDefault();
                            const srcIdx = e.dataTransfer.getData("text/plain");
                            const srcRow = document.querySelector(`[data-idx="${srcIdx}"]`);
                            const tgtRow = e.currentTarget;
                            if (!srcRow || srcRow === tgtRow) return;
                            const srcSpeaker = srcRow.querySelector(".speaker-label");
                            const tgtSpeaker = tgtRow.querySelector(".speaker-label");
                            const tmp = srcSpeaker.textContent;
                            srcSpeaker.textContent = tgtSpeaker.textContent;
                            tgtSpeaker.textContent = tmp;
                            try {
                                await fetch("/segment_update", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({
                                        client_id: getClientId(),
                                        filename: window.currentTranscriptFile || "",
                                        segment_index: parseInt(srcIdx, 10),
                                        new_speaker: srcSpeaker.textContent.replace(/\[|\]/g, "").trim()
                                    })
                                });
                            } catch (err) {
                                console.warn("Failed to persist speaker change:", err);
                            }
                        });
                        box.appendChild(row);
                        partialRow = row;
                    }
                }
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
    // La sauvegarde finale se fait de manière asynchrone côté serveur
    // via 'save_audio_file' lors de la déconnexion de la websocket.
    if (ws) { ws.close(); ws = null; }

    // Ajout d'un indicateur visuel de l'effort de repasse en cours
    const box = document.getElementById('liveTranscript');
    if (box && isRecording && box.innerText.trim() !== "" && !box.innerText.includes("Repasse finale de l'IA en cours")) {
        const row = document.createElement("div");
        row.className = "sentence-row finalized";
        row.style.marginTop = "1rem";
        row.style.color = "var(--text-action-high-blue-france)";
        row.innerHTML = "<strong>⏳ Enregistrement terminé. Traitement final en arrière-plan en cours...</strong>";
        box.appendChild(row);
        box.scrollTop = box.scrollHeight;
    }

    if (window.currentLiveSessionId) {
        localStorage.setItem('pending_job', window.currentLiveSessionId);
        pollStatus(window.currentLiveSessionId);
        window.currentLiveSessionId = null;
    } else {
        setTimeout(loadHistory, 3000);
        setTimeout(loadHistory, 8000);
        setTimeout(loadHistory, 15000);
    }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        if (audioStream._micStream) audioStream._micStream.getTracks().forEach(t => t.stop());
        if (audioStream._sysStream) audioStream._sysStream.getTracks().forEach(t => t.stop());
        audioStream = null;
    }
    processor = null;
    source = null;
    isRecording = false;
    const btnRecord = document.getElementById('recordBtn');
    const btnSystem = document.getElementById('systemBtn');
    const btnMeeting = document.getElementById('meetingBtn');
    btnRecord.innerText = "🎤 Micro";
    btnSystem.innerText = "🖥️ Système";
    btnMeeting.innerText = "👥 Réunion";
    btnRecord.classList.remove('recording');
    btnSystem.classList.remove('recording');
    btnMeeting.classList.remove('recording');
    btnRecord.disabled = false;
    btnSystem.disabled = false;
    btnMeeting.disabled = false;
    document.getElementById('audioBarCont').style.display = 'none';
}

// ========================= HISTORIQUE =========================
async function loadHistory() {
    const list = document.getElementById('transcriptionList');
    try {
        const res = await fetch('/transcriptions/?client_id=' + getClientId());
        const files = await res.json();
        list.innerHTML = "";
        if (files.length === 0) {
            list.textContent = "Aucune.";
            return;
        }

        files.forEach(function (item) {
            const f = item.name;
            const hasCleanup = item.has_cleanup;
            const isBatch = f.startsWith('batch_');
            let audioFilename;
            if (isBatch) {
                const ts = f.replace('batch_', '').replace('.txt', '');
                audioFilename = 'batch_' + getClientId() + '_' + ts + '.wav';
            } else {
                const ts = f.replace('live_', '').replace('.txt', '');
                audioFilename = 'live_' + getClientId() + '_' + ts + '.wav';
            }
            const downloadUrl = '/download_audio/' + getClientId() + '/' + audioFilename;
            const transcriptUrl = '/download_transcript/' + getClientId() + '/' + f;
            const voirLabel = hasCleanup ? "Voir" : "Voir (temporaire)";

            const colDiv = document.createElement('div');
            colDiv.className = 'fr-col-12 fr-col-md-4';

            const cardDiv = document.createElement('div');
            cardDiv.className = 'fr-card fr-card--sm';
            cardDiv.style.padding = '1rem';
            cardDiv.style.border = '1px solid #3a3a3a';

            const title = document.createElement('h4');
            title.className = 'fr-card__title';
            title.style.fontSize = '0.8rem';
            title.textContent = f;

            const viewLink = document.createElement('a');
            viewLink.href = '/view/' + getClientId() + '/' + encodeURIComponent(f);
            viewLink.className = 'fr-btn fr-btn--sm fr-mt-1w';
            viewLink.textContent = voirLabel;

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w';
            deleteBtn.dataset.filename = f;
            deleteBtn.style.color = '#c00';
            deleteBtn.innerHTML = '&#x2716;';
            deleteBtn.onclick = function () { deleteTranscription(this); };

            const downloadAudioLink = document.createElement('a');
            downloadAudioLink.href = downloadUrl;
            downloadAudioLink.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w';
            downloadAudioLink.download = true;
            downloadAudioLink.textContent = 'Audio';

            const rttmLink = document.createElement('a');
            rttmLink.href = '/download_rttm/' + getClientId() + '/' + f;
            rttmLink.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w fr-mr-1w';
            rttmLink.download = true;
            rttmLink.textContent = '💾 RTTM';

            const diarBtn = document.createElement('button');
            diarBtn.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w fr-mr-1w';
            diarBtn.dataset.filename = f;
            diarBtn.textContent = '🔍 Diarisation';
            diarBtn.onclick = function () {
                const clientId = getClientId();
                const filename = this.getAttribute('data-filename');
                window.open("/view_diarization/" + clientId + "/" + encodeURIComponent(filename), "_blank");
            };

            let summaryBtn, actionsBtn;
            if (hasCleanup) {
                summaryBtn = document.createElement('button');
                summaryBtn.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w fr-mr-1w';
                summaryBtn.dataset.filename = f;
                summaryBtn.textContent = '📄 Compte Rendu (Albert)';
                summaryBtn.onclick = function () { generateSummary(this); };

                actionsBtn = document.createElement('button');
                actionsBtn.className = 'fr-btn fr-btn--sm fr-btn--secondary fr-mt-1w fr-mr-1w';
                actionsBtn.dataset.filename = f;
                actionsBtn.textContent = '📝 Actions (Albert)';
                actionsBtn.onclick = function () { generateActions(this); };
            }

            cardDiv.appendChild(title);
            cardDiv.appendChild(viewLink);
            cardDiv.appendChild(document.createTextNode(' '));
            cardDiv.appendChild(downloadAudioLink);
            cardDiv.appendChild(rttmLink);
            cardDiv.appendChild(diarBtn);
            if (hasCleanup) {
                cardDiv.appendChild(summaryBtn);
                cardDiv.appendChild(actionsBtn);
            }
            cardDiv.appendChild(deleteBtn);

            colDiv.appendChild(cardDiv);
            list.appendChild(colDiv);
        });
    } catch (e) {
        console.error("Error loading history:", e);
        list.textContent = "Erreur.";
    }
}
async function deleteTranscription(btn) {
    const filename = btn.getAttribute('data-filename');
    const clientId = getClientId();
    if (!confirm('Confirmer la suppression de ' + filename + ' ?')) return;
    const res = await fetch('/transcriptions/' + encodeURIComponent(filename) + '?client_id=' + clientId, { method: 'DELETE' });
    if (res.ok) {
        loadHistory();
    } else {
        alert('Erreur lors de la suppression');
    }
}
window.deleteTranscription = deleteTranscription;

async function generateSummary(btn) {
    const filename = btn.getAttribute('data-filename');
    const clientId = getClientId();
    window.open("/view_summary/" + clientId + "/" + encodeURIComponent(filename), "_blank");
}
window.generateSummary = generateSummary;


async function generateActions(btn) {
    const filename = btn.getAttribute('data-filename');
    const clientId = getClientId();
    window.open("/view_actions/" + clientId + "/" + encodeURIComponent(filename), "_blank");
}
window.generateActions = generateActions;




// ========================= CONFIGURATION MODELE =========================
function changeModel(e) {
    const model = e.target.value;
    if (!model) return;

    // Afficher/cacher les options Albert selon le modèle sélectionné
    const albertOptions = document.getElementById('albertOptions');
    if (albertOptions) {
        albertOptions.style.display = (model === 'albert') ? 'block' : 'none';
    }

    // Appelle l'API sans authentification forte
    fetch('/change_model?model=' + encodeURIComponent(model), { method: 'POST' })
        .then(res => {
            if (!res.ok) alert("Erreur ou accès refusé lors du changement de modèle");
            else {
                console.log("Modèle changé avec succès.");
                checkRateLimitStatus();
            }
        })
        .catch(err => console.error(err));
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
    const statusElem = document.getElementById('batchStatus');
    for (let i = 0; i < total; i++) {
        if (statusElem) statusElem.innerText = `⏳ Téléchargement morceau ${i + 1}/${total}...`;
        const formData = new FormData();
        formData.append("file_id", id);
        formData.append("client_id", getClientId());
        formData.append("chunk_index", i);
        formData.append("total_chunks", total);
        formData.append("file", file.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE));
        await fetch('/batch_chunk', { method: 'POST', body: formData });
        const pct = Math.round((i + 1) / total * 100);
        const fill = document.getElementById('uploadProgressFill');
        if (fill) {
            fill.style.width = pct + '%';
            fill.style.setProperty('--progress', pct + '%');
        }
        const container = document.getElementById('uploadProgressContainer');
        if (container) container.setAttribute('aria-valuenow', pct);
    }
    pollStatus(id);
}
async function pollStatus(id) {
    let attempts = 0;
    const MAX_ATTEMPTS = 150; // ~5 minutes à 2s d'intervalle
    const interval = setInterval(async () => {
        attempts++;
        if (attempts > MAX_ATTEMPTS) {
            clearInterval(interval);
            localStorage.removeItem('pending_job');
            document.getElementById('uploadBtn').disabled = false;
            document.getElementById('batchStatus').innerText = "⚠️ Timeout : job introuvable ou serveur injoignable.";
            return;
        }
        try {
            const res = await fetch(`/status/${id}`);
            const data = await res.json();
            if (data.status === "terminé") {
                clearInterval(interval);
                localStorage.removeItem('pending_job');
                const uploadBtn = document.getElementById('uploadBtn');
                if (uploadBtn) uploadBtn.disabled = false;
                const statusElem = document.getElementById('batchStatus');
                if (statusElem) {
                    statusElem.innerHTML = '✓ Terminé. <a href="/" class="fr-link">Actualiser</a>';
                }
                loadHistory();
            } else if (data.status === "not_found" || data.status === "erreur") {
                if (data.status === "erreur" && data.error_details) {
                    alert("Erreur lors du traitement : " + data.error_details);
                }
                clearInterval(interval);
                localStorage.removeItem('pending_job');
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('batchStatus').innerText = "";
            } else if (data.status.startsWith("processing:")) {
                const pct = data.progress || 0;
                const eta = data.eta ? " (ETA: " + data.eta + "s)" : '';
                const statusLabel = (data.status && data.status.includes(":")) ? data.status.split(":")[1] : (data.status || "unknown");
                document.getElementById('batchStatus').innerText = "⏳ " + statusLabel + " — " + pct + "%" + eta;
                document.getElementById('uploadBtn').disabled = true;
                const fill = document.getElementById('uploadProgressFill');
                if (fill) {
                    fill.style.width = pct + '%';
                    fill.style.setProperty('--progress', pct + '%');
                }
                const container = document.getElementById('uploadProgressContainer');
                if (container) {
                    container.style.display = 'block';
                    container.setAttribute('aria-valuenow', pct);
                }
            }
        } catch (e) {
            console.warn('pollStatus error:', e);
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
                alert('Résultat de la mise à jour :\n' + (updData.stdout || "Succès."));

                // Afficher un message de redémarrage
                const statusElem = document.getElementById('batchStatus');
                if (statusElem) {
                    statusElem.innerText = "🔄 Le serveur redémarre avec la nouvelle version... Veuillez patienter.";
                    document.getElementById('uploadProgressContainer').style.display = 'block';
                    document.getElementById('uploadProgressFill').style.width = '100%';
                }

                // Attendre que le serveur redémarre (environ 10s)
                setTimeout(() => {
                    location.reload();
                }, 10000);
            }
        }
    } catch (e) {
        console.error('Erreur git status:', e);
    }
}

// ========================= INITIALISATION =========================
window.onload = () => {
    // Initialise les ecouteurs d'evenements
    const elRecord = document.getElementById('recordBtn');
    if (elRecord) elRecord.addEventListener('click', toggleMicrophone);
    const elSystem = document.getElementById('systemBtn');
    if (elSystem) elSystem.addEventListener('click', toggleSystemAudio);
    const elMeeting = document.getElementById('meetingBtn');
    if (elMeeting) elMeeting.addEventListener('click', toggleMeeting);
    const elUpload = document.getElementById('uploadBtn');
    if (elUpload) elUpload.addEventListener('click', handleBatchAction);

    const elToggleSpeaker = document.getElementById('toggleSpeakerEditorBtn');
    if (elToggleSpeaker) elToggleSpeaker.addEventListener('click', toggleSpeakerEditor);

    const elModelSelector = document.getElementById('modelSelector');
    if (elModelSelector) elModelSelector.addEventListener('change', changeModel);

    // Add event listener for audio device selection
    const elAudioDeviceSelect = document.getElementById('audioDeviceSelect');
    if (elAudioDeviceSelect) {
        elAudioDeviceSelect.addEventListener('change', (e) => {
            selectedAudioDeviceId = e.target.value;
            console.log('Selected audio device:', selectedAudioDeviceId);
        });
    }

    // Load available audio devices when page loads
    loadAudioDevices();
    // ---------- Notification de fallback CPU ----------
    (function () {
        // Crée un conteneur de notification s'il n'existe pas
        let fallbackDiv = document.getElementById('fallbackNotice');
        if (!fallbackDiv) {
            fallbackDiv = document.createElement('div');
            fallbackDiv.id = 'fallbackNotice';
            fallbackDiv.style.position = 'fixed';
            fallbackDiv.style.top = '0';
            fallbackDiv.style.left = '0';
            fallbackDiv.style.right = '0';
            fallbackDiv.style.backgroundColor = '#ffcc00';
            fallbackDiv.style.color = '#000';
            fallbackDiv.style.padding = '0.5rem';
            fallbackDiv.style.textAlign = 'center';
            fallbackDiv.style.fontWeight = 'bold';
            fallbackDiv.style.zIndex = '1000';
            fallbackDiv.style.display = 'none';
            document.body.appendChild(fallbackDiv);
        }

        // Fonction de vérification du statut du rate limiter
        async function checkFallbackStatus() {
            try {
                const resp = await fetch('/rate_limiter_status');
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.fallback_active) {
                    const secs = data.fallback_remaining_seconds;
                    fallbackDiv.textContent = `⚠️ Le serveur est en mode de secours CPU pendant ${secs} seconde${secs !== 1 ? 's' : ''} suite à un dépassement de quota.`;
                    fallbackDiv.style.display = 'block';
                } else {
                    fallbackDiv.style.display = 'none';
                }
            } catch (e) {
                console.warn('Erreur lors de la récupération du statut du rate limiter :', e);
            }
        }

        // Vérifie immédiatement puis toutes les 5 s
        checkFallbackStatus();
        setInterval(checkFallbackStatus, 5000);
    })();



    // Chargement initial des donnees
    loadHistory();


    // Initialiser le sélecteur avec le modèle actuel envoyé par le serveur
    const modelDisplay = document.getElementById('currentModelDisplay');
    const modelSelect = document.getElementById('modelSelector');
    if (modelDisplay && modelSelect) {
        const currentModel = modelDisplay.textContent.trim().toLowerCase();
        // Si le modèle actuel est connu, on le sélectionne.
        if (['whisper', 'voxtral', 'albert', 'mock'].includes(currentModel)) {
            if (!modelSelect.disabled) {
                modelSelect.value = currentModel;
            }
        } else {
            // Cas CPU‑only : aucune option GPU disponible, on force Albert.
            if (!modelSelect.disabled && modelSelect.options.length === 1) {
                const onlyOption = modelSelect.options[0].value;
                if (onlyOption === 'albert') {
                    modelSelect.value = 'albert';
                }
            }
        }

        // Initial visibility for Albert options
        const albertOptions = document.getElementById('albertOptions');
        if (albertOptions) {
            albertOptions.style.display = (modelSelect.value === 'albert') ? 'block' : 'none';
        }
    }

    // Verifier les mises a jour Git
    checkGitStatus();

    // Verifier le statut du rate limiter Albert
    checkRateLimitStatus();
    setInterval(checkRateLimitStatus, 10000);

    // Reprendre le suivi d'un batch en cours apres rafraichissement
    const pendingJob = localStorage.getItem('pending_job');

    function clearPendingJob() {
        localStorage.removeItem('pending_job');
        const uploadBtn = document.getElementById('uploadBtn');
        const progressContainer = document.getElementById('uploadProgressContainer');
        const statusElem = document.getElementById('batchStatus');
        if (uploadBtn) uploadBtn.disabled = false;
        if (progressContainer) progressContainer.style.display = 'none';
        if (statusElem) statusElem.innerText = '';
    }

    if (pendingJob) {
        const uploadBtn = document.getElementById('uploadBtn');
        const progressContainer = document.getElementById('uploadProgressContainer');
        const statusElem = document.getElementById('batchStatus');

        // Verifier le statut cote serveur AVANT d'afficher quoi que ce soit
        fetch(`/status/${pendingJob}`)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'terminé') {
                    clearPendingJob();
                    if (statusElem) statusElem.innerText = '✓ Terminé.';
                    loadHistory();
                } else if (data.status === 'not_found' || !data.status || data.status === 'erreur') {
                    if (data.status === 'erreur' && data.error_details) {
                        alert("Erreur lors du traitement : " + data.error_details);
                    }
                    // Job orphelin/erreur : nettoyage
                    clearPendingJob();
                    loadHistory();
                } else if (data.status === 'uploading') {
                    // Un upload ne peut pas être "en cours" au chargement de la page (la boucle JS est morte)
                    clearPendingJob();
                    loadHistory();
                } else {
                    // Job reellement en cours : afficher la progression
                    const pct = data.progress || 0;
                    const eta = data.eta ? ' (ETA: ' + data.eta + 's)' : '';
                    const statusLabel = (data.status && data.status.includes(':')) ? data.status.split(':')[1] : (data.status || 'traitement');

                    if (uploadBtn) uploadBtn.disabled = true;
                    if (progressContainer) {
                        progressContainer.style.display = 'block';
                        progressContainer.setAttribute('aria-valuenow', pct);
                    }
                    const fill = document.getElementById('uploadProgressFill');
                    if (fill) {
                        fill.style.width = pct + '%';
                        fill.style.setProperty('--progress', pct + '%');
                    }
                    if (statusElem) statusElem.innerText = '⏳ ' + statusLabel + ' — ' + pct + '%' + eta;
                    pollStatus(pendingJob);
                }
            })
            .catch(() => {
                // Serveur injoignable : nettoyage silencieux, pas de blocage UI
                clearPendingJob();
            });
    }
};

// ========================= AUDIO DEVICE MANAGEMENT =========================
async function loadAudioDevices() {
    try {
        // Check if mediaDevices API is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            console.warn('MediaDevices API not supported');
            return;
        }

        // Enumerate audio input devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioDevices = devices.filter(device => device.kind === 'audioinput');

        const selectElement = document.getElementById('audioDeviceSelect');
        if (!selectElement) {
            console.warn('Audio device select element not found');
            return;
        }

        // Clear existing options (keep the default option)
        selectElement.innerHTML = '<option value="default">Système par défaut</option>';

        // Add audio devices to the select dropdown
        audioDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Microphone ${device.deviceId.substring(0, 8)}`;
            selectElement.appendChild(option);
        });

        console.log(`Loaded ${audioDevices.length} audio input devices`);
        audioDeviceCount = audioDevices.length;
        console.log('Audio device count:', audioDeviceCount);
    } catch (error) {
        console.error('Error loading audio devices:', error);
        // Even if we fail, keep the default option
        const selectElement = document.getElementById('audioDeviceSelect');
        if (selectElement) {
            selectElement.innerHTML = '<option value="default">Système par défaut</option>';
        }
    }
};
