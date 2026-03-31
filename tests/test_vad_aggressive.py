import pytest

from backend.core.vad import VADManager


@pytest.mark.asyncio
async def test_aggressive_mode_deactivation_on_silence() -> None:
    """
    Vérifie que le mode agressif désactive immédiatement la parole
    lorsqu'un chunk silencieux (aucun octet) est fourni.
    """
    vad = VADManager(aggressive=True)
    silence_chunk = b''  # Aucun cadre, donc silence complet
    result = await vad.check_deactivation(silence_chunk)
    assert result is False
