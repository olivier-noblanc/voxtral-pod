import numpy as np

def decode_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)

def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0

def float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    return (audio_np * 32768.0).astype(np.int16).tobytes()