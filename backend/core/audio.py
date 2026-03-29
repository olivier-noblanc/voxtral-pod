import os
import subprocess
import numpy as np
import ffmpeg

def decode_audio(audio_path: str, sample_rate: int = 16000, timeout: int = 900) -> np.ndarray:
    """
    Decode an audio file to a mono ``float32`` NumPy array using ``ffmpeg-python``.
    
    It works with any audio format that ffmpeg can read (wav, mp3, m4a, flac, …) 
    and resamples to the requested ``sample_rate`` while converting to
    32‑bit floating‑point PCM.

    Parameters
    ----------
    audio_path: str
        Path to the source audio file.
    sample_rate: int, optional
        Desired output sample rate (default 16000 Hz).
    timeout: int, optional
        Timeout for the ffmpeg process in seconds (default 900).

    Returns
    -------
    np.ndarray
        1‑D ``float32`` array containing the decoded audio samples.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Build and run the ffmpeg command using ffmpeg-python
        out, err = (
            ffmpeg.input(audio_path)
            .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True, cmd='ffmpeg')
        )
    except ffmpeg.Error as e:
        # Include ffmpeg stderr for easier debugging
        stderr_msg = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed: {stderr_msg}") from e
    except Exception as e:
        raise RuntimeError(f"An error occurred during audio decoding: {e}") from e

    # Convert raw bytes to NumPy float32 array
    audio = np.frombuffer(out, dtype=np.float32)

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