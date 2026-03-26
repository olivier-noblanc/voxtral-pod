import pytest
from backend.core.vad import VADManager

def test_aggressive_mode_deactivation_on_silence():
    """
    Vérifie que le mode agressif désactive immédiatement la parole
    lorsqu'un chunk silencieux (aucun octet) est fourni.
    """
    vad = VADManager(aggressive=True)
    silence_chunk = b''  # Aucun cadre, donc silence complet
    assert vad.check_deactivation(silence_chunk) is False