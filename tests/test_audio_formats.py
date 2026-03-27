import os
import pathlib
import numpy as np
import pytest
from backend.core.audio import decode_audio

def test_decode_wav_as_m4a_fallback():
    """
    Test que decode_audio gère un fichier même si son extension ne correspond pas,
    en utilisant ffmpeg.
    """
    # Créer un petit fichier WAV de test
    test_wav = "test_fake.m4a"
    import wave, struct
    obj = wave.open(test_wav, 'w')
    obj.setnchannels(1)
    obj.setsampwidth(2)
    obj.setframerate(16000)
    # 1 seconde de silence
    obj.writeframes(struct.pack('<' + 'h'*16000, *([0]*16000)))
    obj.close()

    try:
        # Tenter de charger le fichier (qui est un WAV mais nommé .m4a)
        audio = decode_audio(test_wav, sample_rate=16000)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 16000
        assert audio.dtype == np.float32
    finally:
        if os.path.exists(test_wav):
            os.remove(test_wav)

def test_decode_non_existent():
    with pytest.raises(Exception):
        decode_audio("non_existent_file.mp3")

import shutil
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not found")
def test_decode_real_mp3_if_any_found():
    """
    Si un MP3 existe dans le repo, on essaie de le lire.
    Sinon on skip.
    """
    # Chercher un petit audio existant pour le test
    paths = list(pathlib.Path(".").glob("**/*.mp3")) + list(pathlib.Path(".").glob("**/*.m4a"))
    if not paths:
        pytest.skip("Aucun fichier MP3/M4A trouvé pour le test.")
    
    path = str(paths[0])
    audio = decode_audio(path)
    assert len(audio) > 0
