import numpy as np
import soundfile as sf
import ffmpeg


def decode_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode an audio file to a mono ``float32`` NumPy array using FFmpeg.
    FFmpeg is used as the primary decoder because it supports almost all audio formats
    (MP3, M4A, WAV, FLAC, etc.) and performs resampling in a single pass.
    """
    try:
        # Use ffmpeg to decode and resample in one go
        # f32le: float 32-bit little-endian
        # ac: 1 (mono)
        # ar: resample to target sample_rate
        out, err = (
            ffmpeg.input(audio_path)
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return np.frombuffer(out, dtype=np.float32)
    except Exception as e:
        import sys
        if hasattr(e, "stderr") and e.stderr:
            sys.stderr.write(f"\n[!] FFmpeg DECODE ERROR:\n{e.stderr.decode()}\n")
            sys.stderr.flush()
        
        # Fallback: if ffmpeg fails, try soundfile (legacy behavior for standard WAVs)
        try:
            data, native_sr = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if native_sr != sample_rate:
                # Still need to resample if SR mismatch
                in_bytes = data.tobytes()
                out_res, err_res = (
                    ffmpeg.input("pipe:0", format="f32le", ar=native_sr, ac=1)
                    .output("pipe:0", format="f32le", ar=sample_rate, ac=1)
                    .run(input=in_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
                )
                data = np.frombuffer(out_res, dtype=np.float32)
            return data.astype(np.float32)
        except Exception as fallback_err:
            raise RuntimeError(f"Échec du décodage audio (FFmpeg et SoundFile) : {e} / {fallback_err}")


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