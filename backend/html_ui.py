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
    </style>
</head>
<body>
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

    <dialog id="viewerDialog" role="dialog" aria-labelledby="v-title">
        <div class="fr-container">
            <div class="fr-grid-row fr-grid-row--center">
                <div class="fr-col-12 fr-col-md-10">
                    <div class="fr-modal__body">
                        <div class="fr-modal__header">
                            <button class="fr-link--close fr-link" onclick="closeViewer()">Fermer</button>
                        </div>
                        <div class="fr-modal__content">
                            <h1 id="v-title" class="fr-modal__title">Transcription</h1>
                            <div id="viewerContent" style="white-space: pre-wrap; background: #161616; color: #eee; padding: 1rem; height: 50vh; overflow-y: auto; border: 1px solid #3a3a3a;"></div>
                            <div class="fr-mt-2w fr-p-2w" style="background: #2a2a2a; border-radius: 4px;">
                                <div class="fr-checkbox-group">
                                    <input type="checkbox" id="includeTimestamps" checked onchange="updateExportPreview()">
                                    <label class="fr-label" for="includeTimestamps">Repères temporels</label>
                                </div>
                                <div id="speakerRenameContainer" class="fr-mt-2w" style="display:none;">
                                    <div id="speakerRenameList" class="fr-grid-row fr-grid-row--gutters"></div>
                                </div>
                            </div>
                        </div>
                        <div class="fr-modal__footer">
                            <ul class="fr-btns-group fr-btns-group--inline-reverse">
                                <li><button class="fr-btn" onclick="copyToClipboard()">📋 Copier</button></li>
                                <li><button id="btnSaveS3" class="fr-btn fr-btn--secondary" onclick="uploadToS3()">☁️ S3</button></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </dialog>

    <script>
    // --- Variables globales ---
    let currentText = "";
    let ws, audioContext, source, processor, isRecording = false;
    let audioStream = null;
    let captureType = "mic";
    const CHUNK_SIZE = 4 * 1024 * 1024;

    // --- Utilitaires ---
    const getClientId = () => {
        let id = localStorage.getItem("sota_client_id");
        if (!id) { id = "user_" + crypto.randomUUID().slice(0, 8); localStorage.setItem("sota_client_id", id); }
        return id;
    };

    // --- Modale ---
    async function viewFile(name) {
        const dialog = document.getElementById('viewerDialog');
        if (!dialog) return;
        document.getElementById('v-title').innerText = name;
        dialog.classList.add('fr-modal--opened');
        dialog.showModal();
        const res = await fetch(`/transcription/${name}?client_id=${getClientId()}`);
        currentText = await res.text();
        detectSpeakers(currentText);
        updateExportPreview();
    }

    function closeViewer() {
        const dialog = document.getElementById('viewerDialog');
        if (dialog) {
            dialog.classList.remove('fr-modal--opened');
            dialog.close();
        }
    }

    function detectSpeakers(text) {
        const cont = document.getElementById('speakerRenameList');
        cont.innerHTML = "";
        const matches = text.match(/\[(SPEAKER_\d+)\]/g) || [];
        const speakers = [...new Set(matches)];
        if (speakers.length) {
            document.getElementById('speakerRenameContainer').style.display = "block";
            speakers.forEach((s, index) => {
                const spk = s.slice(1, -1);
                const inputId = `speakerRename_${index}`;
                const div = document.createElement('div');
                div.className = "fr-col-6";
                div.innerHTML = `<label class="fr-label fr-text--xs" for="${inputId}">${spk}</label><input id="${inputId}" name="${inputId}" class="fr-input fr-input--sm" type="text" value="${spk}" data-orig="${spk}" oninput="updateExportPreview()">`;
                cont.appendChild(div);
            });
        } else {
            document.getElementById('speakerRenameContainer').style.display = "none";
        }
    }

    function updateExportPreview() {
        let t = currentText;
        document.querySelectorAll('#speakerRenameList input').forEach(i => {
            t = t.replace(new RegExp('\\[' + i.dataset.orig + '\\]', 'g'), `[${i.value}]`);
        });
        if (!document.getElementById('includeTimestamps').checked) {
            t = t.replace(/\[\d+\.?\d*s -> \d+\.?\d*s\]\s*/g, "");
        }
        document.getElementById('viewerContent').innerText = t;
    }

    function copyToClipboard() {
        navigator.clipboard.writeText(document.getElementById('viewerContent').innerText);
        alert("Copié !");
    }

    // --- Historique ---
    async function loadHistory() {
        const list = document.getElementById('transcriptionList');
        try {
            const res = await fetch(`/transcriptions?client_id=${getClientId()}`);
            const files = await res.json();
            list.innerHTML = files.map(f => `
                <div class="fr-col-12 fr-col-md-4">
                    <div class="fr-card fr-card--sm" style="padding:1rem; border:1px solid #3a3a3a;">
                        <h4 class="fr-card__title" style="font-size:0.8rem">${f}</h4>
                        <button class="fr-btn fr-btn--sm fr-mt-1w" onclick="viewFile('${f}')">Voir</button>
                    </div>
                </div>
            `).join('') || "Aucune.";
        } catch (e) { list.innerHTML = "Erreur."; }
    }

    // --- Changement de modèle ---
    async function changeModel() {
        const newModel = document.getElementById('modelSelector').value;
        const res = await fetch(`/change_model?model=${newModel}`, { method: 'POST' });
        if (res.ok) location.reload();
    }

    // --- Configuration S3 ---
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

    // --- Options Albert ---
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

    // --- Batch upload ---
    async function handleBatchAction() { await uploadFile(); }
    async function uploadFile() {
        const input = document.getElementById('audioFile');
        if (!input.files.length) return;
        const file = input.files[0];
        const total = Math.ceil(file.size / CHUNK_SIZE);
        const id = crypto.randomUUID();
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
                document.getElementById('batchStatus').innerText = "✓ Terminé.";
                loadHistory();
            } else if (data.status.startsWith("processing:")) {
                document.getElementById('batchStatus').innerText = "⏳ " + data.status.split(":")[1];
            }
        }, 2000);
    }

    // --- Gestion du microphone / système (extraits essentiels) ---
    // (Gardez ici le code existant pour `startRecording`, `stopRecording`, etc.
    // Assurez-vous de ne pas dupliquer ces fonctions.)
    // Exemple simplifié : les fonctions existantes doivent être conservées.

    // --- Initialisation ---
    window.onload = () => {
        loadHistory();
        loadS3Config();
        loadAlbertConfig();

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
</body>
</html>
"""
