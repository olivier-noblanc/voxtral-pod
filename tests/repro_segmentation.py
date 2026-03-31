"""
Script de reproduction pour la segmentation Albert.
Valide que les tranches audio sont maximisées.
"""
import os
import sys

import numpy as np

# Ajout du chemin racine pour l'import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.transcription import TranscriptionEngine


def test_repro() -> None:
    """Simule un audio avec silences et vérifie le découpage."""
    engine = TranscriptionEngine(model_id="albert")
    sr = 16000
    # 40 min = 2400s
    duration_total = 2500 
    audio = np.ones(duration_total * sr, dtype=np.float32) * 0.1
    
    # Silence à 10s (devrait être ignoré par la fenêtre de recherche de 3 min)
    audio[int(10 * sr):int(11 * sr)] = 0.0
    
    # Silence à 2390s (devrait être le point de coupe choisi)
    audio[int(2390 * sr):int(2391 * sr)] = 0.0
    
    print("--- REPRO TEST ---")
    print("Total duration: 2500s")
    print("Targeting first cut at: 2400s")
    
    # Appel de la méthode protégée (pylint-disable ok pour un script de repro)
    # pylint: disable=protected-access
    cut = engine._find_best_cut(audio, 2400, current_t=0)
    print(f"Found cut at: {cut:.2f}s")
    
    if cut < 2000:
        print("[!] BUG: Coupe trop précoce (10s au lieu de 2390s).")
    else:
        print("[OK] Coupe optimale (fin de segment privilégiée).")

if __name__ == "__main__":
    test_repro()
