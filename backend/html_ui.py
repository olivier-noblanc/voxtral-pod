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
                        <div class="fr-upséi-group">
                            <input class="fr-upload" type="file" id="audioFile" accept="audio/*,video/*">
                        </div>
<button onclick="handleBatchAction()" id="uploadBtn" class="fr-btn fr-mt-2w" disabled>Transcrire</button>
                        <div id="batchStatus" class="fr-text--bold fr-mt-1w"></div>
                        <div class="fr-progress-bar fr-mt-1w" id="uploadProgressContainer" style="display:none;" role="progressbar">
                            <div class="fr-progress-bar__bar"><div class="fr-progress-bar__fill" id="uploadProgressFill" style="width: 0%"></div></div>
                        </div>
                        <div id="batchResult" class="live-box fr-mt-2w" style="display:none; height: 151px;"></div>
                    </section>
                </div>

                <div class="fr-col-12 fr-mt-4w">
<h3 class="fr-h6">⚙️ Configuration S3 (Optionnel)</h3>
<button id="toggleS3" class="fr-btn fr-btn--secondary fr-mb-2w" onclick="toggleS3Config()">Afficher/masquer S3</button>
<div id="s3Config" class="fr-grid-row fr-grid-row--gutters" style="display:none;">
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

    <script src="/static/app.js"></script>
"""
