import numpy as np
import soundfile as sf
import ffmpeg


def decode_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode an audio file to a mono ``float32`` NumPy array.

    The function first reads the file with ``soundfile`` which natively returns
    ``float32`` data.  If the file's native sample rate differs from the requested
    ``sample_rate`` the audio is resampled using ``ffmpeg‑python``.  The result is
    always a one‑dimensional array (mono) with values in the range ``[-1.0, 1.0]``.
    """
    # Load audio with soundfile – returns data as float32 and the original SR.
    data, native_sr = sf.read(audio_path, dtype="float32")
    # Ensure mono: if the file has multiple channels, average them.
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample only when necessary.
    if native_sr != sample_rate:
        # ``ffmpeg`` works with raw PCM data; we pipe the float32 PCM.
        # Convert NumPy array to raw bytes (little‑endian float32).
        in_bytes = data.tobytes()
        # Run ffmpeg to resample.
        out, _ = (
            ffmpeg.input("pipe:0", format="f32le", ar=native_sr, ac=1)
            .output("pipe:0", format="f32le", ar=sample_rate, ac=1)
            .run(input=in_bytes, capture_stdout=True, capture_stderr=True)
        )
        # Convert the output bytes back to a NumPy array.
        data = np.frombuffer(out, dtype=np.float32)

    return data.astype(np.float32)


def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    """
    Convert 16‑bit PCM (int16) to ``float32`` in the range ``[-1.0, 1.0]``.
    """
    return np.where(pcm >= 0, pcm / 32767.0, pcm / 32768.0).astype(np.float32)


def float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    """
    Convert a ``float32`` audio array (range ``[-1.0, 1.0]``) to 16‑bit PCM bytes.
    Positive values are scaled to ``32767`` and negative values to ``-32768``.
    """
    scaled = np.where(audio_np >= 0, audio_np * 32767.0, audio_np * 32768.0)
    return scaled.astype(np.int16).tobytes()