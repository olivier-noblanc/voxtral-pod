"""
Moteur de diarisation optimisé pour CPU utilisant la bibliothèque 'diarize' (Alexander Lukashov).
Gère le traitement par lots (batch) et le benchmarking des performances (RTF).
"""
import logging
import os
import tempfile
import time
from typing import List, Dict, Any

import soundfile as sf
from diarize import diarize as run_diarize


# Configuration des logs pour la diarisation
logger = logging.getLogger("diarization_cpu")


class LightDiarizationEngine:
    """
    Moteur de diarisation optimisé pour CPU (Xeon) utilisant la bibliothèque 'diarize'.
    Conçu pour le traitement batch de fichiers longs (jusqu'à 3h).
    """
    def __init__(self):
        pass

    def load(self):
        """Optionnel : pré-charge les modèles Silero/WeSpeaker."""
        # La lib diarize charge à la demande, mais on pourrait forcer un warmup ici si besoin.
        pass

    def diarize_file(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Exécute la diarisation sur un fichier audio complet.
        Retourne une liste de segments standardisée.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        logger.info("[*] Début de la diarisation batch : %s", audio_path)
        # run_diarize() retourne un objet DiarizeResult
        result = run_diarize(audio_path)

        segments = []
        for s in result.segments:
            segments.append({
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "speaker": str(s.speaker)
            })

        logger.info("[*] Diarisation terminée : %d segments trouvés.", len(segments))
        return segments

    def benchmark(self, audio_path: str) -> Dict[str, Any]:
        """
        Exécute la diarisation et mesure les performances (RTF, wall clock, etc.).
        """
        start_wall = time.perf_counter()

        # On a besoin de la durée de l'audio pour calculer le RTF
        info = sf.info(audio_path)
        duration = info.duration

        segments = self.diarize_file(audio_path)

        total_time = time.perf_counter() - start_wall
        rtf = total_time / duration if duration > 0 else 0

        # Calcul de la fragmentation (segments par minute)
        frag_per_min = (len(segments) / (duration / 60)) if duration > 0 else 0

        # Calcul du nombre unique de speakers
        speakers = set(s['speaker'] for s in segments)

        stats = {
            "file": os.path.basename(audio_path),
            "duration_sec": round(duration, 2),
            "wall_clock_sec": round(total_time, 2),
            "rtf": round(rtf, 4),
            "segments_count": len(segments),
            "speakers_count": len(speakers),
            "fragmentation_seg_min": round(frag_per_min, 2),
            "segments": segments
        }

        logger.info("[*] Benchmark terminé pour %s : RTF=%s, Segments=%d",
                    stats['file'], stats['rtf'], stats['segments_count'])
        return stats

    def diarize(self, audio_float32, _hook=None):
        """
        Méthode de compatibilité pour le pipeline SotaASR existant.
        Note : l'API 'diarize' originale préfère les chemins de fichiers pour optimiser
        le chargement, mais on peut adapter via un fichier temporaire.
        """
        # On implémente via tempfile si SotaASR l'appelle avec un buffer numpy.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio_float32, 16000)
            return self.diarize_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)