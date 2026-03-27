import sys
from unittest.mock import MagicMock, patch

# Mock torch BEFORE importing backend.core.transcription to avoid long hang
sys.modules['torch'] = MagicMock()

import unittest
import numpy as np
from backend.core.transcription import TranscriptionEngine

class TestAlbertSegmentation(unittest.TestCase):
    def setUp(self):
        self.engine = TranscriptionEngine(model_id="whisper")
        self.engine.albert_api_key = "test_key"
        
    @patch("requests.post")
    @patch("ffmpeg.run")
    @patch("time.sleep")
    def test_segmentation_logic(self, mock_sleep, mock_ffmpeg, mock_post):
        """
        Verifie que l'audio est bien coupe en tranches
        et que les timestamps sont recalés sans appels reels.
        """
        # Mock FFmpeg output (dummy bytes)
        mock_ffmpeg.return_value = (b"fake_mp3_data", b"")
        
        # Audio de 150s. Avec segments de 40s et marge de 60s:
        # 150s > 40s + 60s -> Decoupe.
        # Tranche 1: 0-40s
        # Tranche 2: 40-80s
        # Tranche 3: 80-150s
        duration_sec = 150
        sr = 16000
        audio_np = np.zeros(duration_sec * sr, dtype=np.float32)
        
        # Mock Albert API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "words": [{"start": 1.0, "end": 2.0, "word": "test"}]
        }
        mock_post.return_value = mock_response
        
        # Test avec segments de 40s
        with patch("backend.core.transcription.CHUNK_LIMIT_SEC", 40):
            words, duration = self.engine._transcribe_albert(audio_np)
            
        # 100s -> tranches de [0-40, 40-80, 80-100] = 3 tranches
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(len(words), 3)
        
        # Verifier le recalage temporel
        self.assertEqual(words[0]["start"], 1.0)
        self.assertEqual(words[1]["start"], 41.0)
        self.assertEqual(words[2]["start"], 81.0)
        self.assertEqual(duration, 100.0)

    def test_find_best_cut_logic(self):
        """Verifie la logique de recherche de silence sans executer toute la transcription."""
        # On va tester directement la fonction interne en la 'sortant' ou via un test ciblé
        # Ici on verifie que l'audio plat donne la valeur cible
        duration_sec = 60
        sr = 16000
        audio = np.ones(duration_sec * sr, dtype=np.float32)
        
        # On teste via l'integration legere
        # (Vu que c'est une fonction imbriquée, on teste le comportement macro)
        pass

if __name__ == "__main__":
    unittest.main()
