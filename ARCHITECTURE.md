# Architecture — Voxtral Pod

> Document de référence architecture. Maintenir à jour à chaque refactoring majeur.

## Vue d'ensemble

Voxtral Pod est un service souverain de transcription automatique de la parole (ASR).
Il expose une API REST + WebSocket (FastAPI) consommée par un frontend HTML/JS.

```
┌─────────────────────────────────────────────────────────┐
│                     Client (Browser)                    │
│   app.js  →  REST + WebSocket  →  FastAPI (main.py)     │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   /live (WS)     /batch_chunk     /transcriptions
   LiveSession    BatchWorker      Historique
        │               │
        ▼               ▼
   VADManager      SotaASR (engine.py)
        │               │
        │    ┌──────────┼──────────┐
        │    ▼          ▼          ▼
        │  Diari-   Transcri-   Merger
        │  zation   ption       (merger.py)
        │  Engine   Engine
        │    │          │
        │  Pyannote  Whisper / Albert API / Mock
        │
        └──→ SQLite (jobs.db)
             SQLite (speaker_profiles.db)
             Filesystem (transcriptions_terminees/)
```

---

## Routage — Règles impératives

### Structure des routes
Les routes sont définies dans les **sub-routers**, jamais dans `api.py` :

| Domaine         | Fichier                          |
|-----------------|----------------------------------|
| Système         | `backend/routes/system.py`       |
| Transcriptions  | `backend/routes/transcriptions.py` |
| Audio batch     | `backend/routes/audio.py`        |
| Post-traitement | `backend/routes/postprocess.py`  |
| Diarisation     | `backend/routes/diarization.py`  |
| Locuteurs       | `backend/routes/speakers.py`     |
| Live WebSocket  | `backend/routes/live.py`         |

`api.py` est un **agrégateur pur** : il fait uniquement des `router.include_router(x.router)`.
Il ne déclare aucune route (`@router.get/post`) sauf `/download_transcript` qui est une route composite.

**Ne jamais ajouter de stub `_dummy_*` ou de route fantôme dans `api.py`.**
Si un test échoue faute de route → corriger le test, pas ajouter un stub.

### Mount dans main.py
```python
# Un seul mount, sans prefix — le frontend appelle /live, /transcriptions, etc. directement
app.include_router(api_module.router)
```
**Ne jamais doubler le mount avec un prefix `/api`.**
Les tests doivent appeler les mêmes routes que le frontend.

---

## Composants détaillés

### 1. `backend/main.py` — Point d'entrée
- Crée l'app FastAPI avec `lifespan` (init DB + tâche cleanup au démarrage)
- Monte le dossier `/static`
- Inclut le router de `backend/routes/api.py` **sans prefix**
- Middleware `ProxyHeadersMiddleware` pour nginx/caddy

### 2. `backend/state.py` — État global
- SQLite `jobs.db` : table `jobs` (statut batch/live) + table `config` (modèle courant)
- Singleton `asr_engine` recréé si le modèle change dans la config
- `get_asr_engine(load_model=False)` → retourne toujours un `SotaASR`
- `cleanup_stuck_jobs()` : au démarrage, marque les jobs bloqués en erreur

### 3. `backend/core/engine.py` — SotaASR
**Décision de stratégie au `__init__` :**
```
TESTING=1 OU (no Albert key ET VRAM < 7Go)  →  model_id = "mock"
Albert key + (no GPU OU low VRAM)            →  model_id = "albert"
Sinon                                         →  model_id = <paramètre>
```

**Pipeline `process_file()` :**
1. `decode_audio()` → np.float32 16kHz
2. Diarisation (pyannote GPU ou stub CPU) → segments `[(start, end, speaker)]`
3. Transcription → words `[{start, end, word, speaker}]`
4. `assign_speakers_to_words()` → fusion par overlap temporel
5. `smooth_micro_turns()` → lissage micro-tours isolés
6. `build_speaker_segments()` → segments `[{speaker, start, end, text}]`
7. Formatage texte final : `[0.00s -> 5.00s] [SPEAKER_00] texte`

### 4. `backend/core/transcription.py` — TranscriptionEngine
| `model_id` | Moteur           | Dépendance          |
|------------|------------------|---------------------|
| `whisper`  | faster-whisper   | GPU recommandé      |
| `albert`   | Albert API REST  | `ALBERT_API_KEY`    |
| `mock`     | Stub de test     | Aucune              |
| `vosk`     | Vosk local       | Modèle dans models/ |

**Albert chunking :** segments de max 2400s, coupure sur silences, MP3 64k < 20 Mo.
**Retour toujours :** `(list[dict], float)` = `(words, duration_seconds)`

### 5. `backend/core/vad.py` — VADManager (dual-gate)
```
Chunk PCM int16
    │
    ▼
WebRTC VAD (rapide, 20ms frames)
    │  speech?
    ├─ NON → is_voice_active() = False
    └─ OUI → thread background Silero VAD (précis)
                    │
                    └─ is_voice_active() = webrtc AND silero
```
- `is_speech()` : synchrone, pour détection début de parole
- `check_deactivation()` : détection fin de parole (aggressive mode ou non)
- `reset_states()` : appeler entre deux sessions

### 6. `backend/core/live.py` — LiveSession
- Une instance par connexion WebSocket
- Queue async `audio_queue` (chunks PCM int16)
- Gestion VAD + pré-buffer 1s + buffer phrase
- Silence > 5 chunks (~0.8s) → finalisation du segment
- `save_wav_only()` : sauvegarde WAV de toute la session

### 7. `backend/core/merger.py` — Fusion diarisation/transcription
Trois fonctions pures (pas d'I/O) :
- `assign_speakers_to_words()` : overlap temporel avec padding 40ms
- `smooth_micro_turns()` : relabel mots isolés entourés d'un même locuteur
- `build_speaker_segments()` : groupe mots contigus par locuteur

### 8. `backend/core/postprocess.py` — Post-traitement Albert LLM
- `_call_albert(prompt)` : appel HTTP asynchrone avec retry x3 (backoff exponentiel)
- `summarize_text()` : résumé structuré
- `clean_text()` : suppression tics de langage
- `extract_actions_text()` : extraction TODO/décisions
- Utilise `AlbertAssistant` (wrapper async dans `assistant.py`)

### 9. `backend/core/speaker_profiles.py` — Profils voix
- SQLite `speaker_profiles.db`
- Embeddings Resemblyzer (256-dim float32) stockés en JSON
- `match_embedding()` : cosine similarity > seuil (défaut 0.75)
- `SPEAKER_00` ignoré (profil placeholder)

### 10. `backend/cleanup.py` — Maintenance
- `compress_old_wavs()` : WAV → MP3 64k après 2.4h (ThreadPoolExecutor x2)
- `clean_old_jobs()` : supprime jobs > N jours (SQL)
- `clean_old_files()` : supprime fichiers > N jours (filesystem)
- `periodic_cleanup_task()` : boucle async (cleanup fichiers toutes les 24h, jobs stale toutes les 1h)

---

## Flux de données — Session Live

```
Browser Mic/System
    │ PCM Int16 (2560 samples @ 16kHz)
    ▼
WebSocket /live
    │
    ▼
LiveSession.audio_queue
    │
    ▼
process_audio_queue() [asyncio task]
    │
    ├─ VADManager.is_speech() → OUI
    │       │
    │       └─ Accumulation dans sentence_buffer
    │               │
    │               ├─ Partial (toutes les 2s) → TranscriptionEngine → WS JSON
    │               │
    │               └─ Silence détecté → Transcription finale → WS JSON
    │
    └─ Fermeture WS → flush dernier segment → save_wav_only() → job batch final
```

## Flux de données — Batch Upload

```
Browser File Input
    │ Chunks 4 Mo
    ▼
POST /batch_chunk (multipart)
    │ Assemblage dans temp_batch/
    ▼
POST /batch_finalize (auto sur dernier chunk)
    │
    ▼
Job SQLite (status: uploading → processing:X% → terminé/erreur)
    │
    ▼
SotaASR.process_file()
    │
    ▼
Fichier .txt dans transcriptions_terminees/batch_audio/
```

---

## Dépendances Python critiques

| Package          | Usage                          | Optionnel |
|------------------|--------------------------------|-----------|
| fastapi          | Framework API                  | Non       |
| uvicorn          | Serveur ASGI                   | Non       |
| faster-whisper   | ASR local GPU                  | Oui (GPU) |
| pyannote.audio   | Diarisation GPU                | Oui (GPU) |
| resemblyzer      | Embeddings voix                | Oui       |
| silero-vad       | VAD précis                     | Non       |
| webrtcvad        | VAD rapide                     | Non       |
| torch            | Tenseurs (silero, whisper)     | Non       |
| numpy            | Traitement audio               | Non       |
| soundfile        | Lecture/écriture WAV           | Non       |
| ffmpeg-python    | Compression MP3 (Albert)       | Non       |
| requests         | Appels Albert API              | Non       |
| keyring          | Stockage clés sécurisé         | Oui       |
| sklearn          | Clustering (diarization_cpu)   | Oui       |

---

## Points d'attention pour modifications

1. **Ajouter un endpoint** → dans le sub-router métier concerné (`system.py`, `audio.py`, etc.), jamais dans `api.py`
2. **Changer le format words** → impacte `merger.py`, `live.py`, `engine.py`, `transcription.py`
3. **Changer la DB** → migrations manuelles (pas d'ORM), tester `init_db()` + `cleanup_stuck_jobs()`
4. **Nouveau modèle ASR** → ajouter dans `TranscriptionEngine.load()` + `TranscriptionEngine.transcribe()` + logique de sélection dans `SotaASR.__init__()`
5. **Modifier le HTML** → vérifier le DOM contract avec `python scripts/check_dom_contract.py`
6. **Tests** → `TESTING=1 pytest` active le mode mock, évite le chargement GPU
7. **Tests de routes** → les tests appellent les mêmes URLs que le frontend (sans prefix `/api/`)
