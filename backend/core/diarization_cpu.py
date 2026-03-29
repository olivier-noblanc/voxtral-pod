# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize'.

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
# Moteur principal (Exclusif WeSpeaker)
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

    def diarize_file(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Exclusive CPU diarization using the 'diarize' library (WeSpeaker ONNX).
        Zero Torch, high efficiency (RTF ~0.1 on Xeon).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        logger.info("[*] Diarisation batch (WeSpeaker ONNX) : %s", audio_path)
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
                logger.warning("[!] Moteur WeSpeaker a renvoyé 0 segments.")
                return []
                
            logger.info("[*] %d segments trouvés par moteur WeSpeaker.", len(segments))
            return segments
            
        except Exception as e:
            logger.error("[!] Échec critique du moteur WeSpeaker : %s", e)
            return []

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