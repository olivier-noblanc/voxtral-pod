# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize'.
Patch : téléchargement du modèle WeSpeaker via miroir HuggingFace
        au lieu de Tencent Cloud (bloqué par proxy d'entreprise).

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
from typing import List, Dict, Any

import soundfile as sf

logger = logging.getLogger("diarization_cpu")

# ---------------------------------------------------------------------------
# Patch WeSpeaker : redirection du téléchargement vers HuggingFace
# ---------------------------------------------------------------------------
# Le modèle est publié sur HF par l'équipe WeSpeaker :
# https://huggingface.co/wenet-e2e/wespeaker-voxceleb-resnet34-LM
_WESPEAKER_MODEL_DIR = Path.home() / ".wespeaker" / "en"
_WESPEAKER_MODEL_FILE = _WESPEAKER_MODEL_DIR / "voxceleb_resnet34_LM.onnx"

_HF_MIRROR_URL = (
    "https://huggingface.co/openspeech/wespeaker-models"
    "/resolve/main/voxceleb_resnet34_LM.onnx"
)


def _ensure_wespeaker_model() -> None:
    """
    Vérifie que le modèle WeSpeaker est présent localement.
    Si absent, le télécharge depuis HuggingFace (pas Tencent Cloud).
    Thread-safe via fichier .lock simple.
    """
    if _WESPEAKER_MODEL_FILE.exists():
        return

    _WESPEAKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _WESPEAKER_MODEL_FILE.with_suffix(".lock")

    # Évite les téléchargements concurrents (multi-worker)
    if lock_path.exists():
        logger.info("[*] Modèle WeSpeaker en cours de téléchargement par un autre process, attente...")
        for _ in range(60):
            time.sleep(2)
            if _WESPEAKER_MODEL_FILE.exists():
                return
        raise RuntimeError("Timeout attente modèle WeSpeaker (.lock présent trop longtemps)")

    try:
        lock_path.touch()
        logger.info("[*] Téléchargement modèle WeSpeaker depuis HuggingFace...")
        logger.info(f"    URL  : {_HF_MIRROR_URL}")
        logger.info(f"    Dest : {_WESPEAKER_MODEL_FILE}")

        tmp_path = _WESPEAKER_MODEL_FILE.with_suffix(".tmp")

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                if pct % 10 == 0:
                    logger.info(f"    ... {pct}%")

        urllib.request.urlretrieve(_HF_MIRROR_URL, tmp_path, reporthook=_progress)
        tmp_path.rename(_WESPEAKER_MODEL_FILE)
        logger.info("[*] Modèle WeSpeaker téléchargé avec succès.")

    except Exception as e:
        # Nettoyage en cas d'erreur
        for p in [tmp_path if 'tmp_path' in dir() else None, lock_path]:
            if p and Path(p).exists():
                Path(p).unlink(missing_ok=True)
        raise RuntimeError(f"Échec téléchargement modèle WeSpeaker depuis HuggingFace : {e}") from e
    finally:
        lock_path.unlink(missing_ok=True)


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
        # Pré-télécharge le modèle au démarrage si nécessaire
        # (évite une erreur 403 en cours de traitement)
        _ensure_wespeaker_model()

    def load(self) -> None:
        """Optionnel : warmup explicite de la lib diarize."""
        pass

    def diarize_file(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Diarise un fichier audio complet.
        Retourne une liste de segments [{start, end, speaker}].
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        from diarize import diarize as run_diarize

        logger.info("[*] Diarisation batch : %s", audio_path)
        result = run_diarize(audio_path)

        segments = [
            {
                "start": round(float(s.start), 3),
                "end":   round(float(s.end),   3),
                "speaker": str(s.speaker),
            }
            for s in result.segments
        ]
        logger.info("[*] %d segments trouvés.", len(segments))
        return segments

    def benchmark(self, audio_path: str) -> Dict[str, Any]:
        """RTF + métriques de performance."""
        start_wall = time.perf_counter()
        info = sf.info(audio_path)
        duration = info.duration

        segments = self.diarize_file(audio_path)

        elapsed = time.perf_counter() - start_wall
        rtf = elapsed / duration if duration > 0 else 0
        speakers = {s["speaker"] for s in segments}

        return {
            "file":                   os.path.basename(audio_path),
            "duration_sec":           round(duration, 2),
            "wall_clock_sec":         round(elapsed, 2),
            "rtf":                    round(rtf, 4),
            "segments_count":         len(segments),
            "speakers_count":         len(speakers),
            "fragmentation_seg_min":  round(len(segments) / (duration / 60), 2) if duration > 0 else 0,
            "segments":               segments,
        }

    def diarize(self, audio_float32, hook=None) -> List[Dict[str, Any]]:
        """
        Compatibilité pipeline SotaASR (entrée numpy float32).
        Passe par un fichier temporaire WAV 16kHz.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio_float32, 16000)
            return self.diarize_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
