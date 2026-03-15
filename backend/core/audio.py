import subprocess
import numpy as np

def decode_audio(audio_path, sample_rate=16000):
    """
    Decode audio file to float32 mono PCM using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-i", audio_path, "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sample_rate), "-"
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
    
    return np.frombuffer(result.stdout, dtype=np.float32).copy()

def pcm_to_float32(pcm_bytes):
    """Convert int16 PCM bytes to float32 numpy array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def float32_to_pcm16(audio_np):
    """Convert float32 numpy array to int16 PCM bytes."""
    return (audio_np * 32768.0).astype(np.int16).tobytes()
