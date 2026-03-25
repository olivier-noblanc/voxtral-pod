HTML_UI = r"""<!DOCTYPE html>
<html lang="fr" data-fr-scheme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Voxtral Pod — SOTA ASR</title>
    <link rel="stylesheet" href="/static/dsfr.min.css">
    <style>
        :root { --fr-border-default-grey: #3a3a3a; }
        .live-box {
            height: 400px; overflow-y: auto;
            background: #161616; padding: 1.5rem;
            border-radius: 4px; font-family: monospace; color: #10B981;
            border: 1px solid var(--fr-border-default-grey);
        }
        .audio-bar-container {
            width: 100%; height: 8px; background: #3a3a3a;
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
        .sentence-row.finalized { border-left-color: #3b82f6; }
        .partial-text { color: #9ca3af; font-style: italic; }
        .final-text { color: #f9fafb; }
        #recordBtn.recording {
            background-color: #fcebe8 !important;
            color: #e1000f !important;
            box-shadow: inset 0 0 0 1px #e1000f !important;
            animation: pulse-red 1.5s infinite;
        }
        @keyframes pulse-red { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        .fr-header__service-tagline { color: var(--text-mention-grey); }
        /* Corrections pour la modale DSFR */
        dialog.fr-modal[open] {
            display: flex;
        }
        dialog.fr-modal {
            padding: 0;
        }
        dialog.fr-modal .fr-modal__inner {
            max-height: 90vh;
            overflow-y: auto;
        }
        @media (max-width: 768px) {
            dialog.fr-modal .fr-modal__inner {
                max-height: 100vh;
            }
            dialog.fr-modal .fr-modal__body {
                border-radius: 0;
            }
        }
    </style>
</head>
<body data-commit="{{commit}}">
    <header role="banner" class="fr-header">
        <div class="fr-header__body">
            <div class="fr-container">
                <div class="fr-header__body-row">
                    <div class="fr-header__brand">
                        <div class="fr-header__brand-top">
                            <div class="fr-header__logo">
                                <p class="fr-logo">République<br>Française</p>
                            </div>
                        </div>
                        <div class="fr-header__service">
                            <a href="/" title="Accueil - Voxtral Pod">
                                <p class="fr-header__service-title">Voxtral Pod</p>
                            </a>
                            <p class="fr-header__service-tagline">Service de Transcription Automatique</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main role="main" id="content">
        <div class="fr-container fr-py-4w">
            <div class="fr-grid-row fr-grid-row--gutters">
                <div class="fr-col-12">
                    <h1 class="fr-h1">🎙️ Console de Transcription</h1>
                    <div class="fr-highlight fr-mb-3w">
                        <p class="fr-text--sm">
                            GPU: <span class="fr-badge fr-badge--info fr-badge--no-icon">{{device}}</span> | 
                            Modèle: <span class="fr-badge fr-badge--success fr-badge--no-icon" id="currentModelDisplay">{{model_name}}</span>
                        </p>
                    </div>
                    
                    <div class="fr-select-group">
                        <label class="fr-label" for="modelSelector">Modèle ASR</label>
                        <select class="fr-select" id="modelSelector" onchange="changeModel()">
                            <option value="whisper">Faster-Whisper Large-v3</option>
                            <option value="voxtral">Voxtral Mini 4B</option>
                            <option value="albert">API Albert (Étalab)</option>
                        </select>
                        <p id="cpuWarning" class="fr-hint-text fr-mt-1w" style="display:none; color:#ce0500;">
                            ⚠️ Aucun GPU détecté — Albert API forcé automatiquement
                        </p>
                    </div>

                    <div id="albertOptions" class="fr-mt-2w" style="display:none;">
                        <div class="fr-checkbox-group">
                            <input type="checkbox" id="albertPartial" onchange="saveAlbertConfig()">
                            <label class="fr-label" for="albertPartial">
                                Transcriptions partielles (Albert)
                                <span class="fr-hint-text">Note : Cela augmente la consommation de jetons API.</span>
                            </label>
                        </div>
                    </div>
                </div>

                <div class="fr-col-12 fr-col-md-7">
                    <section class="fr-pt-2w">
                        <h2 class="fr-h4">Direct</h2>
                        <div class="fr-btns-group fr-btns-group--inline-md fr-btns-group--sm fr-my-2w">
                            <button id="recordBtn" class="fr-btn" onclick="toggleMicrophone()">🎤 Micro</button>
                            <button id="systemBtn" class="fr-btn fr-btn--secondary" onclick="toggleSystemAudio()">💻 Système</button>
                        </div>
                        <div class="audio-bar-container" id="audioBarCont"><div id="audioBar"></div></div>
                        <div class="live-box" id="liveTranscript">En attente de flux audio...</div>
                    </section>
                </div>

                <div class="fr-col-12 fr-col-md-5">
                    <section class="fr-pt-2w">
                        <h2 class="fr-h4">Fichiers (Batch)</h2>
                        <div class="fr-upload-group">
                            <input class="fr-upload" type="file" id="audioFile" accept="audio/*,video/*">
                        </div>
                        <button onclick="handleBatchAction()" id="uploadBtn" class="fr-btn fr-mt-2w">Transcrire</button>
                        <div id="batchStatus" class="fr-text--bold fr-mt-1w"></div>
                        <div class="fr-progress-bar fr-mt-1w" id="uploadProgressContainer" style="display:none;" role="progressbar">
                            <div class="fr-progress-bar__bar"><div class="fr-progress-bar__fill" id="uploadProgressFill" style="width: 0%"></div></div>
                        </div>
                        <div id="batchResult" class="live-box fr-mt-2w" style="display:none; height: 151px;"></div>
                    </section>
                </div>

                <div class="fr-col-12 fr-mt-4w">
                    <h3 class="fr-h6">⚙️ Configuration S3 (Optionnel)</h3>
                    <div class="fr-grid-row fr-grid-row--gutters">
                        <div class="fr-col-12 fr-col-md-6">
                            <input class="fr-input" type="text" id="s3Endpoint" placeholder="Endpoint" onchange="saveS3Config()">
                        </div>
                        <div class="fr-col-12 fr-col-md-6">
                            <input class="fr-input" type="text" id="s3Bucket" placeholder="Bucket" onchange="saveS3Config()">
                        </div>
                        <div class="fr-col-12 fr-col-md-6">
                            <input class="fr-input" type="password" id="s3AccessKey" placeholder="Access Key" onchange="saveS3Config()">
                        </div>
                        <div class="fr-col-12 fr-col-md-6">
                            <input class="fr-input" type="password" id="s3SecretKey" placeholder="Secret Key" onchange="saveS3Config()">
                        </div>
                    </div>
                </div>

                <div class="fr-col-12 fr-mt-4w">
                    <h2 class="fr-h4">📜 Historique</h2>
                    <div id="transcriptionList" class="fr-grid-row fr-grid-row--gutters fr-mt-2w">Chargement...</div>
                </div>
            </div>
        </div>
    </main>

    <footer class="fr-footer fr-mt-4w" role="contentinfo">
        <div class="fr-container">
            <div class="fr-footer__body">
                <div class="fr-footer__brand">
                    <p class="fr-logo">République<br>Française</p>
                    <p class="fr-footer__brand-title">Voxtral Pod</p>
                </div>
                <div class="fr-footer__content">
                    <p class="fr-footer__content-desc">Solution souveraine de transcription ASR.</p>
                </div>
            </div>
        </div>
    </footer>

    <script>
    // ========================= VARIABLES GLOBALES =========================
    let currentText = "";
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
            const partialAlbert = document.getElementById('albertPartial').checked;
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
        } catch (e) { list.innerHTML = "Erreur."; }
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
                document.getElementById('batchStatus').innerText = "✓ Terminé.";
                loadHistory();
            } else if (data.status === "not_found") {
                clearInterval(interval);
                localStorage.removeItem('pending_job');
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

        // Reprendre un job en cours si présent
        const pendingJob = localStorage.getItem('pending_job');
        if (pendingJob) {
            document.getElementById('uploadProgressContainer').style.display = 'block';
            document.getElementById('batchStatus').innerText = '⏳ Reprise du job en cours...';
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
    </script>
    <script>
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
    </script>
</body>
</html>
"""
