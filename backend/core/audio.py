import os
import subprocess
import numpy as np

def decode_audio(audio_path: str, sample_rate: int = 16000, timeout: int = 300) -> np.ndarray:
    """
    Decode an audio file to a mono ``float32`` NumPy array.

    This implementation uses the ``ffmpeg`` command‑line tool directly,
    avoiding the optional ``ffmpeg-python`` wrapper.  It works with any
    audio format that ffmpeg can read (wav, mp3, m4a, flac, …) and
    resamples to the requested ``sample_rate`` while converting to
    32‑bit floating‑point PCM.

    Parameters
    ----------
    audio_path: str
        Path to the source audio file.
    sample_rate: int, optional
        Desired output sample rate (default 16000 Hz).
    timeout: int, optional
        Timeout for the ffmpeg process in seconds (default 300).

    Returns
    -------
    np.ndarray
        1‑D ``float32`` array containing the decoded audio samples.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output files (not used here)
        "-i", audio_path,         # input file
        "-f", "f32le",            # output format: raw 32‑bit float little‑endian
        "-acodec", "pcm_f32le",   # codec
        "-ac", "1",               # mono
        "-ar", str(sample_rate),  # resample
        "pipe:1",                 # write to stdout
    ]

    try:
        # Run ffmpeg and capture raw PCM data
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {timeout} seconds.") from e
    except subprocess.CalledProcessError as e:
        # Include ffmpeg stderr for easier debugging
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode(errors='ignore')}") from e

    # Convert raw bytes to NumPy float32 array
    audio = np.frombuffer(result.stdout, dtype=np.float32)

    # Ensure the array is one‑dimensional (mono)
    if audio.ndim != 1:
        audio = audio.ravel()

    return audio

def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    """
    Convert int16 PCM data to float32 in the range [-1.0, 1.0].

    Parameters
    ----------
    pcm : np.ndarray
        1‑D array of int16 audio samples.

    Returns
    -------
    np.ndarray
        Float32 array with values normalized to [-1.0, 1.0].
    """
    # Ensure the input is int16
    pcm = pcm.astype(np.int16)
    # Use 32767 for positive max, 32768 for negative min, then clip
    float_arr = pcm.astype(np.float32) / 32767.0
    # Correct the max value edge case
    float_arr = np.where(pcm == 32767, 1.0, float_arr)
    return np.clip(float_arr, -1.0, 1.0)

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """
    Convert a float32 audio array (range [-1.0, 1.0]) to 16‑bit PCM bytes.

    Parameters
    ----------
    audio : np.ndarray
        1‑D float32 array.

    Returns
    -------
    bytes
        PCM16 data as a bytes object.
    """
    # Clip to the valid range to avoid overflow
    audio_clipped = np.clip(audio, -1.0, 1.0)
    # Scale to int16 range and convert
    pcm16 = (audio_clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()