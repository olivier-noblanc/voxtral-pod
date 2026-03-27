"""
Tests de qualité pour la segmentation Albert.
Vérifie que les tranches audio sont optimisées par rapport au silence.
"""
import unittest
import os
import sys
import numpy as np

# Ajout du chemin racine pour l'import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.transcription import TranscriptionEngine

class TestAlbertSegmentationQuality(unittest.TestCase):
    """Vérifie que la segmentation maximise la durée des tranches."""

    def setUp(self):
        """Initialisation du moteur de transcription."""
        self.engine = TranscriptionEngine(model_id="albert")
        self.sr = 16000

    def test_find_best_cut_prefers_late_silence(self):
        """Vérifie que le moteur préfère couper au silence le plus tardif."""
        # 40 min = 2400s
        duration_total = 2500 
        audio = np.ones(duration_total * self.sr, dtype=np.float32) * 0.1
        
        # Silence à 10s (trop tôt, devrait être ignoré par la fenêtre de recherche)
        audio[int(10 * self.sr):int(11 * self.sr)] = 0.0
        
        # Silence à 2390s (idéal, car proche de la limite de 2400s)
        audio[int(2390 * self.sr):int(2391 * self.sr)] = 0.0
        
        # Appel de la fonction de coupe (méthode protégée autorisée dans les tests)
        # pylint: disable=protected-access
        cut = self.engine._find_best_cut(audio, 2400, current_t=0)
        
        # On attend une coupe proche de 2390s
        self.assertGreater(cut, 2000, "La coupe est trop précoce (silence de 10s utilisé au lieu de 2390s).")
        self.assertAlmostEqual(cut, 2390.8, delta=1.0)

    def test_find_best_cut_fallback_to_target(self):
        """Vérifie que le moteur coupe à la target si aucun silence n'est trouvé."""
        duration_total = 2500
        # Audio sans silence (bruit blanc)
        audio = np.random.uniform(-1, 1, duration_total * self.sr).astype(np.float32)
        
        # pylint: disable=protected-access
        cut = self.engine._find_best_cut(audio, 2400, current_t=0)
        
        # Devrait couper dans la fenêtre de recherche [2220, 2400]
        # Dans du bruit blanc, n'importe quel point est "le moins bruyant"
        self.assertGreaterEqual(cut, 2220)
        self.assertLessEqual(cut, 2400)

if __name__ == "__main__":
    unittest.main()
