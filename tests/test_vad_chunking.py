from unittest.mock import MagicMock, patch

import numpy as np


def test_run_silero_chunk_sizes():
    """Vérifie que _run_silero padde correctement tous les chunks à 512."""
    
    received_sizes = []
    
    def fake_silero(tensor, sample_rate):
        received_sizes.append(tensor.shape[0])
        assert tensor.shape[0] == 512, f"Silero reçu {tensor.shape[0]} au lieu de 512"
        mock_result = MagicMock()
        mock_result.item.return_value = 0.5
        return mock_result

    # Patch les imports lourds
    with patch.dict('sys.modules', {
        'webrtcvad': MagicMock(),
        'silero_vad': MagicMock(load_silero_vad=MagicMock(return_value=fake_silero)),
        'torch': MagicMock(from_numpy=lambda x: x),
    }):
        from backend.core.vad import VADManager

        vad = VADManager.__new__(VADManager)
        vad.silero_model = fake_silero
        vad.silero_sensitivity = 0.4
        vad._lock = __import__('asyncio').Lock()

        for size in [512, 853, 1024, 1537, 100]:
            received_sizes.clear()
            pcm = (np.zeros(size, dtype=np.float32) * 32768).astype(np.int16).tobytes()
            result = vad._run_silero(pcm)
            assert all(s == 512 for s in received_sizes), \
                f"Taille {size}: chunks reçus = {received_sizes}"
            assert isinstance(result, float)