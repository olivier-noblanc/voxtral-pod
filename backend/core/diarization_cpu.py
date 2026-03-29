# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize'.
Patch : pré-téléchargement depuis onnx-community HuggingFace (public, sans token)
        dans ~/.wespeaker/en/model.onnx avant que wespeakerruntime ne tente Tencent.

Ref académique :
- Wang et al., "WeSpeaker: Learning Speaker Embeddings with Self-Supervision",
  ICASSP 2023 — https://arxiv.org/abs/2210.02596
"""
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger("diarization_cpu")

# ---------------------------------------------------------------------------
# Pré-téléchargement modèle WeSpeaker
# onnx-community/wespeaker-voxceleb-resnet34-LM — public, CC-BY-4.0, sans token
# Le fichier est déjà nommé model.onnx dans le repo → chemin exact attendu par hub.py
# ---------------------------------------------------------------------------
_HF_MIRROR_URL = (
    "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/onnx/model.onnx"
)
_MODEL_DIR  = Path.home() / ".wespeaker" / "en"
_MODEL_PATH = _MODEL_DIR / "model.onnx"


def _ensure_model() -> None:
    """
    Vérifie que ~/.wespeaker/en/model.onnx existe.
    Si absent, le télécharge depuis HuggingFace (public, sans token).
    wespeakerruntime/hub.py vérifie ce chemin avant tout accès réseau :
    si le fichier est là, Tencent Cloud n'est jamais contacté.
    """
    if _MODEL_PATH.exists():
        logger.info("[*] Modèle WeSpeaker déjà présent : %s", _MODEL_PATH)
        return

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _MODEL_PATH.with_suffix(".tmp")

    logger.info("[*] Téléchargement modèle WeSpeaker depuis HuggingFace...")
    logger.info("    URL  : %s", _HF_MIRROR_URL)
    logger.info("    Dest : %s", _MODEL_PATH)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            if pct % 10 == 0:
                logger.info("    ... %d%%", pct)

    try:
        urllib.request.urlretrieve(_HF_MIRROR_URL, tmp_path, reporthook=_progress)
        tmp_path.rename(_MODEL_PATH)
        logger.info("[*] Modèle WeSpeaker prêt : %s", _MODEL_PATH)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Échec téléchargement WeSpeaker : {e}") from e


# Exécuté à l'import — avant tout appel à wespeakerruntime
_ensure_model()


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------

class LightDiarizationEngine:
    """
    Moteur de diarisation optimisé CPU (Xeon) — lib 'diarize' + WeSpeaker ONNX.
    - Zéro torch
    - Multi-thread via ONNX Runtime (OMP_NUM_THREADS / MKL_NUM_THREADS)
    - RTF ~0.1 sur Xeon (INTERSPEECH 2022 / WeSpeaker ICASSP 2023)
    """

    def __init__(self):
        pass

    def load(self) -> None:
        """Contract placeholder to match DiarizationEngine interface."""
        pass

    def _fallback_diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Robust fallback using Silero VAD + SpeechBrain (ECAPA-TDNN) + HAC.
        Guarantees work because VAD and SpeechBrain are confirmed working with Torch CPU.
        """
        logger.info("[*] SpeechBrain Diarization Fallback initialisée...")
        from backend.core.vad import VADManager
        from backend.core.speaker_manager import SpeakerManager
        import torch

        # Load audio at 16kHz using project's robust utility
        from backend.core.audio import decode_audio
        audio_np = decode_audio(audio_path)
        sr = 16000

        # 1. Segment via VAD
        vad = VADManager(silero_sensitivity=0.5, aggressive=False)
        chunk_size = 512 # Silero chunk
        probs = []
        
        # We process in chunks to find speech
        logger.info("    - Étape 1: Segmentation VAD...")
        segments_raw = []
        is_speech = False
        speech_start = 0
        
        # Simple turn detection
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Use Silero model directly for speed here
            # We copy() the chunk to ensure it is writable and avoid PyTorch UserWarnings.
            prob = vad.silero_model(torch.from_numpy(chunk.copy()), sr).item()
            time_pos = i / sr
            
            if prob > 0.3 and not is_speech: # Slightly more sensitive (0.3 instead of 0.4)
                is_speech = True
                speech_start = time_pos
            elif prob < 0.1 and is_speech: # More robust closing (0.1 instead of 0.2)
                is_speech = False
                if time_pos - speech_start > 0.3: # Min duration 0.3s instead of 0.5s
                    segments_raw.append((speech_start, time_pos))
        
        if is_speech: # close last segment
            segments_raw.append((speech_start, len(audio_np)/sr))

        if not segments_raw:
            logger.warning("[!] Aucun segment de parole détecté par le fallback.")
            return []

        # 2. Extract Embeddings
        logger.info("    - Étape 2: Extraction d'empreintes vocales (SpeechBrain)...")
        spk_manager = SpeakerManager()
        embeddings = []
        valid_segments = []
        
        for start, end in segments_raw:
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            chunk_audio = audio_np[s_idx:e_idx]
            
            try:
                emb = spk_manager.get_embedding(chunk_audio)
                embeddings.append(emb)
                valid_segments.append((start, end))
            except Exception as e:
                logger.debug(f"      Can't extract embedding for {start:.1f}-{end:.1f}: {e}")

        if not embeddings:
            return []

        # 3. Clustering (HAC)
        logger.info("    - Étape 3: Clustering des locuteurs (HAC)...")
        embeddings_matrix = np.stack(embeddings)
        
        # If only one segment or very few, don't cluster
        if len(embeddings_matrix) < 2:
            speaker_labels = np.array([1])
        else:
            try:
                # Use scipy linkage for Agglomerative Clustering (HAC)
                from scipy.spatial.distance import pdist
                # Cosine distance matrix
                dist_matrix = pdist(embeddings_matrix, metric='cosine')
                # Complete linkage is robust for speaker clustering
                Z = linkage(dist_matrix, method='complete')
                # Threshold ~0.25 is typical for normalized speaker embeddings (SpeechBrain/WeSpeaker)
                # This will automatically estimate the number of speakers based on the threshold
                speaker_labels = fcluster(Z, t=0.35, criterion='distance')
            except Exception as e:
                logger.error(f"      Clustering failed: {e}. Fallback to single speaker.")
                speaker_labels = np.ones(len(embeddings_matrix), dtype=int)

        # 4. Final Formatting
        final_segments = []
        for i, (start, end) in enumerate(valid_segments):
            # speaker_labels starts at 1, map to 0-indexed
            label = int(speaker_labels[i]) - 1
            final_segments.append({
                "start":   round(start, 3),
                "end":     round(end, 3),
                "speaker": f"SPEAKER_{label:02d}"
            })

        logger.info("[*] Diarisation Fallback terminée : %d segments avec %d locuteurs trouvés.", 
                    len(final_segments), len(np.unique(speaker_labels)))
        return final_segments

    def diarize_file(self, audio_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        logger.info("[*] Diarisation batch (Primaire) : %s", audio_path)
        try:
            from diarize import diarize as run_diarize
            result = run_diarize(audio_path)
            
            segments = [
                {
                    "start":   round(float(s.start), 3),
                    "end":     round(float(s.end),   3),
                    "speaker": str(s.speaker),
                }
                for s in result.segments
            ]
            
            if not segments:
                logger.warning("[!] Moteur Primaire a renvoyé 0 segments. Tentative du Fallback...")
                return self._fallback_diarize(audio_path)
                
            logger.info("[*] %d segments trouvés par moteur Primaire.", len(segments))
            return segments
            
        except Exception as e:
            logger.error("[!] Échec du moteur Primaire diarize : %s. Basculement Fallback.", e)
            return self._fallback_diarize(audio_path)

    def benchmark(self, audio_path: str) -> Dict[str, Any]:
        start_wall = time.perf_counter()
        info = sf.info(audio_path)
        duration = info.duration

        segments = self.diarize_file(audio_path)

        elapsed = time.perf_counter() - start_wall
        rtf = elapsed / duration if duration > 0 else 0
        speakers = {s["speaker"] for s in segments}

        return {
            "file":                  os.path.basename(audio_path),
            "duration_sec":          round(duration, 2),
            "wall_clock_sec":        round(elapsed, 2),
            "rtf":                   round(rtf, 4),
            "segments_count":        len(segments),
            "speakers_count":        len(speakers),
            "fragmentation_seg_min": round(len(segments) / (duration / 60), 2) if duration > 0 else 0,
            "segments":              segments,
        }

    def diarize(self, audio_float32, hook=None) -> List[Dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio_float32, 16000)
            return self.diarize_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)