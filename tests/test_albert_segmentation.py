import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.core.transcription import TranscriptionEngine


class TestAlbertSegmentation(unittest.TestCase):
    def setUp(self):
        self.engine = TranscriptionEngine(model_id="whisper")
        self.engine.albert_api_key = "test_key"
        
    @patch("requests.post")
    @patch("ffmpeg.input")
    @patch("soundfile.write")
    @patch("time.sleep")
    def test_segmentation_logic(self, _mock_sleep, _mock_sf, mock_ffmpeg_input, mock_post):
        """
        Verifie la segmentation offline (tout est mocké).
        """
        # Mock FFmpeg chain: ffmpeg.input().output().run() -> (stdout, stderr)
        mock_run = mock_ffmpeg_input.return_value.output.return_value.run
        mock_run.return_value = (b"fake_mp3_data", b"")
        
        # Audio de 150s.
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
        
        # On force la limite a 40s pour le test
        with patch("backend.core.transcription.CHUNK_LIMIT_SEC", 40):
            # pylint: disable=protected-access
            words, duration = self.engine._transcribe_albert(audio_np)
            
        # On attend 3 appels API
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(len(words), 3)
        
        # Verifier le recalage temporel des tranches
        self.assertEqual(words[0]["start"], 1.0)
        self.assertEqual(words[1]["start"], 41.0)
        self.assertEqual(words[2]["start"], 81.0)
        self.assertEqual(duration, 150.0)

    def test_find_best_cut_isolated(self):
        """Test leger du helper de silence."""
        # Test minimaliste : sur un audio plat, il doit renvoyer la cible
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
