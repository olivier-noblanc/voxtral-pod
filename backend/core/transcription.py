"""
Transcription motor for Voxtral Pod.
Handles Whisper (local), Vosk (local) and Albert API (remote).
"""
import io
import json
import os
import time
import warnings
from typing import Any, Optional

import numpy as np
import requests
import soundfile as sf

from backend.core.albert_rate_limiter import albert_rate_limiter
from backend.core.postprocess import _ensure_ffmpeg

# Propriété utilitaire : nom des modèles qui utilisent l'API Albert
_ALBERT_MODEL_IDS = frozenset({"albert"})

# Limit de segmentation pour l'API Albert (300s = 50 min)
# On reste sous 20 Mo grâce à la compression MP3 48k (50 min ≈ 17,2 Mo)
# Limite de taille du payload (en Mo) – on vise < 20 Mo
MAX_PAYLOAD_MB = 19
# Chunk limit disabled – aucune contrainte de durée fixe ; la segmentation s’appuie sur la taille du payload
# Nous limitons la durée maximale d’un chunk à 40 minutes (2400 s). 
# Le bitrate est ajusté à 64 kbit/s afin que le MP3 reste < 20 Mo même pour la durée maximale.
CHUNK_LIMIT_SEC = 2400

# Additional fix for KaldiRecognizer cleanup issue
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*KaldiRecognizer")

# Ensure NNPACK is disabled for transcription operations
os.environ.setdefault("DISABLE_NNPACK", "1")

# Global flag to ensure the patch is only applied once
_VOSK_PATCH_APPLIED = False

def _apply_vosk_patch() -> None:
    """
    Patches KaldiRecognizer.__del__ to prevent AttributeError: 'KaldiRecognizer' object has no attribute '_handle'.
    """
    global _VOSK_PATCH_APPLIED
    if _VOSK_PATCH_APPLIED:
        return
    
    try:
        import vosk
        # Ensure we have the class and the original __del__
        if hasattr(vosk, "KaldiRecognizer"):
            original_del = getattr(vosk.KaldiRecognizer, "__del__", None)
            
            def patched_del(self: Any) -> None:
                try:
                    if hasattr(self, "_handle") and self._handle is not None:
                        if original_del:
                            original_del(self)
                except Exception as e:
                    # Log cleanup error sparingly
                    import logging
                    logging.getLogger("transcription").debug(f"Error in KaldiRecognizer cleanup: {e}")
            
            vosk.KaldiRecognizer.__del__ = patched_del
            _VOSK_PATCH_APPLIED = True
    except (ImportError, AttributeError):
        # Vosk not installed or different version
        _VOSK_PATCH_APPLIED = True # Don't try again

class TranscriptionEngine:
    """
    Multi-motor transcription orchestrator.
    """
    def __init__(self, model_id: str = "whisper", device: str | None = None) -> None:
        self.model_id = model_id

        # Automatic device detection (lazy import de torch)
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Any = None
        from backend.config import get_albert_api_key
        self.albert_api_key = get_albert_api_key()
        self.albert_base_url = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
        self.albert_model_id = os.getenv("ALBERT_MODEL_ID", "openai/whisper-large-v3")

    @property
    def use_albert(self) -> bool:
        """True si l'on doit passer par l'API Albert pour la transcription."""
        return self.model_id in _ALBERT_MODEL_IDS

    def load(self) -> None:
        """Charge les modèles en mémoire (Whisper ou Vosk)."""
        if self.use_albert or self.model_id == "mock":
            print(f"[*] Transcription engine in {self.model_id} mode (no local weights needed).")
            return

        if self.model_id == "whisper" or self.model_id == "cpu":
            # Si le modèle est cpu, on force le device à cpu
            device_to_use = "cpu" if self.model_id == "cpu" else self.device
            compute_to_use = "int8" if self.model_id == "cpu" else ("float16" if device_to_use == "cuda" else "int8")
            
            print(f"[*] Loading Faster-Whisper (large-v3) on {device_to_use}...")
            import faster_whisper
            self.model = faster_whisper.WhisperModel(
                "large-v3",
                device=device_to_use,
                compute_type=compute_to_use,
            )
        else:
            import vosk
            print(f"[*] Loading Vosk model: {self.model_id}...")
            self.model = vosk.Model(f"models/{self.model_id}")

    def transcribe(
        self,
        audio_np: np.ndarray,
        language: str = "fr",
        progress_callback: Any = None,
        is_partial: bool = False,
        job_id: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], float, bool, Optional[dict[str, Any]]]:
        """
        Transcribe audio and return (words, duration, used_fallback, raw_data).
        """
        # Tâche 1 : Optimisation des transcriptions partielles (CPU pur)
        if is_partial and self.model_id != "mock":
            use_local_model = True
        else:
            use_local_model = not self.use_albert or self.model_id == "mock"

        used_fallback = False
        if (
            self.use_albert and 
            not use_local_model and 
            albert_rate_limiter.should_use_cpu_fallback_mode()
        ):
            print("[*] Switching to local model (CPU fallback) due to rate limit")
            used_fallback = True
            use_local_model = True
            
            # Switch definitively to CPU mode on fallback so we can load it.
            self.model_id = "cpu"
            if self.model is None:
                self.load()
        
        if use_local_model:
            if self.model_id == "mock":
                duration = len(audio_np) / 16000
                msg = "[MOCK] Transcription de test mélangée (Micro + Système)"
                return [{"start": 0.0, "end": duration, "word": msg}], duration, False, None

            if self.model is None:
                self.load()

            duration = len(audio_np) / 16000

            if self.model_id == "whisper" or self.model_id == "cpu":
                segments, info = self.model.transcribe(
                    audio_np,
                    beam_size=5,
                    language=language,
                    task="transcribe",
                    word_timestamps=True,
                )

                all_words = []
                for s in segments:
                    if progress_callback and duration > 0:
                        pct = int((s.end / duration) * 50)
                        progress_callback(f"Transcription ({int(s.end)}s / {int(duration)}s)", 45 + pct)

                    if hasattr(s, "words") and s.words:
                        for w in s.words:
                            all_words.append({
                                "start": w.start,
                                "end": w.end,
                                "word": w.word,
                                "speaker": "UNKNOWN"
                            })
                    else:
                        all_words.append({
                            "start": s.start,
                            "end": s.end,
                            "word": s.text.strip(),
                            "speaker": "UNKNOWN"
                        })

                return all_words, info.duration, used_fallback, None

            # Vosk
            import vosk
            _apply_vosk_patch()
            
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)

            pcm16 = np.clip(audio_np * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            rec.AcceptWaveform(pcm16)
            res = json.loads(rec.FinalResult())

            all_words = []
            if "result" in res:
                for w in res["result"]:
                    all_words.append({
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"],
                        "speaker": "UNKNOWN"
                    })

            return all_words, len(audio_np) / 16000, used_fallback, None

        # Utiliser l'API Albert
        res_words, res_dur, res_raw = self._transcribe_albert(
            audio_np, language, progress_callback, job_id=job_id
        )
        return res_words, res_dur, False, res_raw

    def _find_best_cut(self, audio_np: np.ndarray, target_sec: float, current_t: float = 0) -> float:
        """
        Trouve le meilleur point de coupe (silence) près de target_sec.
        On recherche dans une fenêtre réduite à la fin du segment (180s)
        afin de maximiser la durée de la tranche.
        """
        sr = 16000
        search_range_sec = 180
        start_sec = max(current_t, target_sec - search_range_sec)
        end_sec = min(len(audio_np) / sr, target_sec)
        
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)

        segment = audio_np[start_idx:end_idx]
        if len(segment) == 0:
            return target_sec

        frame_len = int(sr * 0.2)
        num_frames = len(segment) // frame_len
        if num_frames == 0:
            return target_sec

        frames = segment[:num_frames * frame_len].reshape(num_frames, frame_len)
        energies = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

        SILENCE_THRESHOLD = 1e-3
        silent_indices = np.where(energies < SILENCE_THRESHOLD)[0]
        if silent_indices.size > 0:
            last_silent = silent_indices[-1]
            cut_sec = start_sec + (last_silent * frame_len) / sr
            if cut_sec >= target_sec - 0.5 or cut_sec <= start_sec + 0.5:
                return float(target_sec)
            return float(cut_sec)

        min_idx = np.argmin(energies)
        cut_sec = start_sec + (min_idx * frame_len) / sr
        if cut_sec >= target_sec - 0.5 or cut_sec <= start_sec + 0.5:
            return float(target_sec)
        return float(cut_sec)

    def _transcribe_albert(
        self,
        audio_np: np.ndarray,
        language: str = "fr",
        progress_callback: Any = None,
        is_partial: bool = False,
        job_id: Optional[str] = None
    ) -> tuple[list[dict[str, Any]], float, Optional[dict[str, Any]]]:
        """
        Découpe l'audio en tranches et les envoie à l'API Albert.
        Gère le fractionnement automatique (split) en cas d'échec sur une grosse tranche.
        """
        duration_total = len(audio_np) / 16000

        if albert_rate_limiter.is_in_mock_mode():
            print("[*] Using mock mode for Albert (quota exceeded)")
            return [{"start": 0.0, "end": duration_total, "word": "[MOCK MODE] Quota exceeded."}], duration_total, None

        initial_tranches: list[tuple[float, float]] = []
        t: float = 0.0
        while t < duration_total:
            remaining = duration_total - t
            if remaining <= CHUNK_LIMIT_SEC + 60:
                initial_tranches.append((t, remaining))
                break
            cut_point = self._find_best_cut(audio_np, t + CHUNK_LIMIT_SEC, current_t=int(t))
            initial_tranches.append((t, cut_point - t))
            t = cut_point

        if job_id:
            from backend.state import append_job_log
            append_job_log(job_id, f"Planification : {len(initial_tranches)} tranches (max {CHUNK_LIMIT_SEC}s).")

        all_final_words: list[dict[str, Any]] = []
        all_final_segments: list[dict[str, Any]] = []
        queue = initial_tranches.copy()
        processed_count = 0
        total_planned = len(queue)

        while queue:
            start_time, duration = queue.pop(0)
            if progress_callback:
                prog_base = 45 + int((processed_count / max(1, total_planned)) * 40)
                progress_callback(f"Albert ({processed_count+1}/...)", prog_base)

            chunk_audio = audio_np[int(start_time * 16000): int((start_time + duration) * 16000)]
            import ffmpeg
            _ensure_ffmpeg()
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, chunk_audio, 16000, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)

            try:
                out, _ = (
                    ffmpeg
                    .input('pipe:0', format='wav')
                    .output('pipe:1', format='mp3', audio_bitrate='64k')
                    .run(input=wav_buffer.read(), capture_stdout=True, capture_stderr=True, quiet=True)
                )
                buffer = io.BytesIO(out)
                size_mb = len(out) / (1024*1024)
            except Exception as e:
                print(f"[!] Erreur compression tranche {processed_count+1}: {e}")
                raise e

            if job_id:
                from backend.state import append_job_log
                append_job_log(job_id, f"Tranche @{start_time:.1f}s ({duration:.1f}s, {size_mb:.2f}Mo) - Envoi...")

            headers = {"Authorization": f"Bearer {self.albert_api_key}"}
            files = {"file": ("audio.mp3", buffer, "audio/mpeg")}
            data = {"model": self.albert_model_id, "language": language, "response_format": "verbose_json"}
            
            response = None
            success = False
            last_err = None
            
            for attempt in range(3):
                try:
                    if not albert_rate_limiter.can_make_request():
                        time.sleep(albert_rate_limiter.min_interval_seconds)
                    buffer.seek(0)
                    response = requests.post(
                        f"{self.albert_base_url}/audio/transcriptions", 
                        headers=headers, 
                        files=files, 
                        data=data, 
                        timeout=1800
                    )
                    albert_rate_limiter.record_request()
                    albert_rate_limiter.handle_response(response.status_code)
                    if response.status_code == 200:
                        success = True
                        break
                    if response.status_code == 429:
                        time.sleep(int(response.headers.get("Retry-After", 2 ** (attempt + 1))))
                    elif response.status_code >= 500:
                        time.sleep(2 ** attempt)
                    else:
                        break
                except Exception as e:
                    last_err = e
                    time.sleep(2 ** attempt)

            if not success or response is None:
                status = response.status_code if response is not None else "N/A"
                if duration > 600:
                    if job_id:
                        from backend.state import append_job_log
                        append_job_log(job_id, f"  └─ Échec ({status}). Split en cours...")
                    mid = duration / 2
                    queue.insert(0, (start_time + mid, duration - mid))
                    queue.insert(0, (start_time, mid))
                    total_planned += 1
                    buffer.close()
                    continue

                detail = response.text[:200] if response is not None else str(last_err)
                raise Exception(f"Échec Albert Tranche {processed_count+1} ({status}): {detail}")

            result = response.json()
            chunk_words: list[dict[str, Any]] = []
            if result.get("words"):
                chunk_words = [
                    {"start": w["start"], "end": w["end"], "word": w["word"], "speaker": "UNKNOWN"}
                    for w in result["words"]
                ]
            elif result.get("segments"):
                for s in result["segments"]:
                    if s.get("words"):
                        for w in s["words"]: 
                            chunk_words.append({
                                "start": w["start"], "end": w["end"], "word": w["word"], "speaker": "UNKNOWN"
                            })
                    else:
                        chunk_words.append({
                            "start": s.get("start", 0.0), 
                            "end": s.get("end", duration), 
                            "word": s.get("text", "").strip(), 
                            "speaker": "UNKNOWN"
                        })
            
            if result.get("segments"):
                for s in result["segments"]:
                    all_final_segments.append({
                        "start": s.get("start", 0.0) + start_time,
                        "end": s.get("end", duration) + start_time,
                        "text": s.get("text", "").strip()
                    })
            elif result.get("text"):
                all_final_segments.append({
                    "start": start_time,
                    "end": start_time + duration,
                    "text": result["text"].strip()
                })

            for w in chunk_words:
                all_final_words.append({
                    "start": w["start"] + start_time,
                    "end": w["end"] + start_time,
                    "word": w["word"],
                    "speaker": "UNKNOWN"
                })
            
            buffer.close()
            processed_count += 1

        raw_payload = {"type": "albert_batch", "segments": all_final_segments, "words": all_final_words}
        return all_final_words, duration_total, raw_payload
