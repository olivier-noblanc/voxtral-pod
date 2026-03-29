# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize'.
Patch : interception globale de urllib pour rediriger Tencent Cloud → HuggingFace.

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
# Interception globale urllib — redirige Tencent Cloud → HuggingFace
# La lib diarize/wespeaker hard-code l'URL COS Tencent.
# On remplace urlretrieve pour intercepter AVANT qu'elle tente la connexion.
# ---------------------------------------------------------------------------
_TENCENT_DOMAIN = "cos.ap-shanghai.myqcloud.com"
_HF_TOKEN       = os.getenv("HF_TOKEN", "")
_HF_MIRROR_URL  = (
    "https://huggingface.co/wenet-e2e/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/voxceleb_resnet34_LM.onnx"
)

_original_urlretrieve = urllib.request.urlretrieve


def _patched_urlretrieve(url, filename=None, reporthook=None, data=None):
    if _TENCENT_DOMAIN in url:
        logger.info("[*] WeSpeaker URL interceptée : %s", url)
        logger.info("[*] Redirection → HuggingFace : %s", _HF_MIRROR_URL)

        if _HF_TOKEN:
            opener = urllib.request.build_opener()
            opener.addheaders = [("Authorization", f"Bearer {_HF_TOKEN}")]
            urllib.request.install_opener(opener)
            logger.info("[*] HF_TOKEN injecté dans la requête.")

        return _original_urlretrieve(_HF_MIRROR_URL, filename, reporthook, data)

    return _original_urlretrieve(url, filename, reporthook, data)


# Patch global — actif dès l'import du module
urllib.request.urlretrieve = _patched_urlretrieve
logger.info("[*] Patch urllib WeSpeaker actif (Tencent → HuggingFace).")


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