# backend/core/diarization_cpu.py
"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize' (Alexander Lukashov).
Gère le traitement par lots (batch) et le benchmarking des performances (RTF).
Intègre SpeakerManager pour l'identification automatique des locuteurs.
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

# Ensure NNPACK is disabled for diarization CPU operations
os.environ.setdefault("DISABLE_NNPACK", "1")

logger = logging.getLogger("diarization_cpu")

# ---------------------------------------------------------------------------
# Pré-téléchargement modèle WeSpeaker
# ---------------------------------------------------------------------------
_HF_MIRROR_URL = (
    "https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/onnx/model.onnx"
)
_MODEL_DIR  = Path.home() / ".wespeaker" / "en"
_MODEL_PATH = _MODEL_DIR / "model.onnx"


def _get_r_proxy_settings() -> Optional[dict[str, str]]:
    """Essaye de lire les paramètres de proxy depuis le fichier .Rprofile si présent."""
    try:
        rprofile_path = Path.home() / "Documents" / ".Rprofile"
        if not rprofile_path.exists():
            return None
            
        content = rprofile_path.read_text(encoding='utf-8')
        
        # Chercher les options de téléchargement curl
        curl_match = re.search(r'download\.file\.extra\s*=\s*"(.*?)"', content, re.IGNORECASE)
        if curl_match:
            extra_args = curl_match.group(1)
            # Extraire les paramètres de proxy
            proxy_match = re.search(r'--proxy\s+([^\s]+)', extra_args)
            if proxy_match:
                proxy_url = proxy_match.group(1)
                return {
                    'proxy_url': proxy_url,
                    'extra_args': extra_args
                }
        return None
    except Exception:
        return None


def _download_with_r_proxy(url: str, dest_path: Path, proxy_settings: dict[str, str]) -> None:
    """Télécharge un fichier en utilisant les paramètres de proxy R avec curl."""
    try:
        # Construire la commande curl avec les paramètres R
        cmd = [
            "curl",
            "-L",  # Suivre les redirections
            "--location",
            "--output", str(dest_path),
            url
        ]
        
        # Ajouter les arguments supplémentaires depuis R
        if proxy_settings.get('extra_args'):
            extra_args = proxy_settings['extra_args'].split()
            cmd.extend(extra_args)
        
        logger.info(f"[*] Exécution de la commande curl : {' '.join(cmd)}")
        curl_path = shutil.which("curl")
        if curl_path:
            cmd[0] = curl_path
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
        if result.returncode != 0:
            raise Exception(f"curl failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Erreur lors du téléchargement avec curl: {e.stderr}") from e
    except Exception as e:
        # Si curl échoue, fallback à urllib avec gestion des proxies Windows
        logger.warning(f"[*] Fallback à urllib.request en raison de l'erreur: {e}")
        _download_with_urllib_windows(url, dest_path)


def _download_with_urllib_windows(url: str, dest_path: Path) -> None:
    """Télécharge un fichier en utilisant urllib avec prise en charge des proxies Windows."""
    try:
        # Utiliser urllib avec les proxies système Windows
        import urllib.request
        
        # Créer un gestionnaire de requête avec les proxies système
        proxy_handler = urllib.request.ProxyHandler(urllib.request.getproxies())
        opener = urllib.request.build_opener(proxy_handler)
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, str(dest_path))  # noqa: S310
    except Exception as e:
        # Si tout échoue, essayer sans proxy
        logger.warning(f"[*] Tentative de téléchargement sans proxy: {e}")
        urllib.request.urlretrieve(url, str(dest_path))  # noqa: S310


def _ensure_model() -> None:
    """Vérifie et télécharge le modèle WeSpeaker si nécessaire."""
    if _MODEL_PATH.exists():
        return

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = _MODEL_PATH.with_suffix(".tmp")

    logger.info("[*] Téléchargement modèle WeSpeaker depuis HuggingFace...")
    try:
        # Essayer d'utiliser les paramètres de proxy R si disponibles
        proxy_settings = _get_r_proxy_settings()
        if proxy_settings:
            logger.info("[*] Utilisation des paramètres de proxy R pour le téléchargement...")
            _download_with_r_proxy(_HF_MIRROR_URL, tmp_path, proxy_settings)
        else:
            # Fallback à la méthode originale
            urllib.request.urlretrieve(_HF_MIRROR_URL, tmp_path)  # noqa: S310
        tmp_path.rename(_MODEL_PATH)
        logger.info("[*] Modèle WeSpeaker prêt.")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Échec téléchargement WeSpeaker : {e}") from e


# Initialisation au chargement
_ensure_model()


class LightDiarizationEngine:
    """
    Moteur de diarisation optimisé CPU (Xeon) — WeSpeaker ONNX.
    Conçu pour être rapide (RTF ~0.1) et efficace.
    """

    def __init__(self) -> None:
        pass

    def load(self) -> None:
        """Contract placeholder to match DiarizationEngine interface."""
        pass

    def diarize_file(self, audio_path: str) -> list[dict[str, Any]]:
        """
        Diarisation CPU pure utilisant la bibliothèque 'diarize'.
        Retourne une liste de dictionnaires {start, end, speaker}.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        logger.info("[*] Diarisation batch (WeSpeaker ONNX) : %s", audio_path)
        try:
            from diarize import diarize as run_diarize  # type: ignore[import-untyped]
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
            
        except Exception as e:
            logger.error("[!] Échec moteur diarize : %s", e)
            return []

    def benchmark(self, audio_path: str) -> dict[str, Any]:
        """Mesure de performance (RTF)."""
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

    def diarize(self, audio_float32: np.ndarray, hook: Any = None) -> list[tuple[float, float, str]]:
        """
        Interface principale pour SotaASR.process_file().
        IMPORTANT : Doit renvoyer une liste de TUPLES (start, end, speaker).
        Effectue également la RECONNAISSANCE des locuteurs via SpeakerManager.
        """
        from backend.core.speaker_manager import SpeakerManager
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Écrire le buffer audio
            sf.write(tmp_path, audio_float32, 16000)
            
            # 1. Diarisation
            raw_segments = self.diarize_file(tmp_path)
            
            # 2. Identification (matching avec profils enrôlés)
            manager = SpeakerManager()
            identified_segments = manager.identify_speakers_in_segments(tmp_path, raw_segments)
            
            # 3. Conversion en TUPLES pour merger.py
            return [
                (float(s["start"]), float(s["end"]), str(s["speaker"]))
                for s in identified_segments
            ]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)