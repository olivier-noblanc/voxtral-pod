import os
import tempfile
import wave

import numpy as np
import pytest

# Skip the entire test module if ffmpeg is not installed
pytest.importorskip("ffmpeg")

# Functions under test are defined in backend.core.audio
from backend.core.audio import decode_audio, float32_to_pcm16, pcm_to_float32


def _write_wav(file_path: str, audio_np: np.ndarray, sample_rate: int = 16000) -> None:
    """
    Helper to write a mono WAV file using the standard library ``wave`` module.
    The input array is expected to be float32 in the range [-1.0, 1.0].
    """
    # Convert to int16 PCM for WAV storage (using 32768.0 normalization consistent with core)
    pcm16 = np.clip(audio_np * 32768.0, -32768, 32767).astype(np.int16)
    # Open a Wave_write object to write the synthetic audio.
    # Adding a type hint helps static analysers (and silences pylint warnings).
    wf = wave.open(file_path, "wb")  # type: ignore[assignment]
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 2 bytes = 16 bits
    wf.setframerate(sample_rate)
    wf.writeframes(pcm16.tobytes())
    wf.close()


def test_decode_audio_returns_float32() -> None:
    """
    Generate a synthetic sine wave, write it to a temporary WAV file,
    decode it with ``decode_audio`` and verify the output dtype and shape.
    """
    duration_sec = 0.1
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "sine.wav")
        _write_wav(wav_path, sine, sample_rate=sr)

        decoded = decode_audio(wav_path, sample_rate=sr)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        # Shape should be (N,) where N = sr * duration_sec
        assert decoded.shape == sine.shape
        # Values should be close to the original (allow small ffmpeg conversion error)
        np.testing.assert_allclose(decoded, sine, atol=1e-4)


def test_pcm_to_float32_normalization() -> None:
    """
    Verify that the conversion from int16 PCM to float32 normalizes correctly.
    """
    max_int16 = np.int16(32767)
    min_int16 = np.int16(-32768)

    max_float = pcm_to_float32(np.array([max_int16], dtype=np.int16))
    min_float = pcm_to_float32(np.array([min_int16], dtype=np.int16))

    # Expected values are approximately 1.0 (actually 32767/32768) and -1.0
    assert np.isclose(max_float[0], 32767/32768, atol=1e-7)
    assert np.isclose(min_float[0], -1.0, atol=1e-7)


def test_float32_roundtrip() -> None:
    """
    Convert a random float32 array to PCM16 and back, ensuring the
    round‑trip error stays below 1e-4.
    """
    rng = np.random.default_rng(42)
    original = rng.uniform(-1.0, 1.0, size=1024).astype(np.float32)

    pcm_bytes = float32_to_pcm16(original)
    recovered = pcm_to_float32(np.frombuffer(pcm_bytes, dtype=np.int16))

    # The round‑trip error should be very small
    diff = np.abs(original - recovered)
    assert np.max(diff) < 1e-4