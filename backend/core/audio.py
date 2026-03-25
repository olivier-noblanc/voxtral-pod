import numpy as np

def decode_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)

def pcm_to_float32(pcm: np.ndarray) -> np.ndarray:
    return np.where(pcm >= 0, pcm / 32767.0, pcm / 32768.0).astype(np.float32)

def float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    # Scale positive to 32767 and negative to 32768
    scaled = np.where(audio_np >= 0, audio_np * 32767.0, audio_np * 32768.0)
    return scaled.astype(np.int16).tobytes()