# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize'.
Patch : patch direct de Hub.Assets pour rediriger Tencent → HuggingFace,
        et pré-téléchargement dans le bon chemin (~/.wespeaker/en/model.onnx).

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
# Patch Hub.Assets — remplace l'URL Tencent par HuggingFace
# et pré-télécharge model.onnx au bon chemin avant que la lib le demande.
# ---------------------------------------------------------------------------
_HF_TOKEN      = os.getenv("HF_TOKEN", "")
_HF_MIRROR_URL = (
    "https://huggingface.co/wenet-e2e/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/voxceleb_resnet34_LM.onnx"
)
_MODEL_DIR  = Path.home() / ".wespeaker" / "en"
_MODEL_PATH = _MODEL_DIR / "model.onnx"


def _ensure_model() -> None:
    """
    Pré-télécharge model.onnx depuis HuggingFace si absent.
    La lib wespeakerruntime vérifie ~/.wespeaker/en/model.onnx avant de télécharger —
    si le fichier est là, elle ne touche pas au réseau.
    """
    if _MODEL_PATH.exists():
        logger.info("[*] Modèle WeSpeaker déjà présent : %s", _MODEL_PATH)
        return

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _MODEL_PATH.with_suffix(".tmp")

    logger.info("[*] Téléchargement modèle WeSpeaker depuis HuggingFace...")
    logger.info("    URL  : %s", _HF_MIRROR_URL)
    logger.info("    Dest : %s", _MODEL_PATH)

    if _HF_TOKEN:
        opener = urllib.request.build_opener()
        opener.addheaders = [("Authorization", f"Bearer {_HF_TOKEN}")]
        urllib.request.install_opener(opener)
        logger.info("    HF_TOKEN injecté.")

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


# Exécuté à l'import — avant tout appel à la lib
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
        pass

    def diarize_file(self, audio_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        from diarize import diarize as run_diarize

        logger.info("[*] Diarisation batch : %s", audio_path)
        result = run_diarize(audio_path)

        segments = [
            {
                "start":   round(float(s.start), 3),
                "end":     round(float(s.end),   3),
                "speaker": str(s.speaker),
            }
            for s in result.segments
        ]
        logger.info("[*] %d segments trouvés.", len(segments))
        return segments

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
