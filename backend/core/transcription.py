import os
import io

import numpy as np


# Propriété utilitaire : nom des modèles qui utilisent l'API Albert
_ALBERT_MODEL_IDS = frozenset({"albert"})


# Limite de segmentation pour l'API Albert (3600s = 1h)
# On reste sous 25Mo grace a la compression MP3 48k (1h ~= 21.6Mo)
CHUNK_LIMIT_SEC = 3600

class TranscriptionEngine:
    def __init__(self, model_id="whisper", device=None):
        self.model_id = model_id

        # Automatic device detection (lazy import de torch)
        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self.model = None
        self.albert_api_key = os.getenv("ALBERT_API_KEY")
        self.albert_base_url = "https://albert.api.etalab.gouv.fr/v1"
        self.albert_model_id = os.getenv("ALBERT_MODEL_ID", "openai/whisper-large-v3")

    @property
    def use_albert(self) -> bool:
        """True si l'on doit passer par l'API Albert pour la transcription."""
        return self.model_id in _ALBERT_MODEL_IDS or bool(self.albert_api_key)

    def load(self):
        if self.use_albert or self.model_id == "mock":
            print(f"[*] Transcription engine in {self.model_id} mode (no local weights needed).")
            return

        if self.model_id == "whisper":
            print(f"[*] Loading Faster-Whisper (large-v3) on {self.device}...")
            compute_type = "float16" if self.device == "cuda" else "int8"
            # Lazy import of faster_whisper
            import faster_whisper
            self.model = faster_whisper.WhisperModel(
                "large-v3",
                device=self.device,
                compute_type=compute_type,
            )
        else:
            # Import Vosk uniquement quand nécessaire (lib lourde)
            # Lazy import of vosk
            import vosk  # noqa: PLC0415
            print(f"[*] Loading Vosk model: {self.model_id}...")
            self.model = vosk.Model(f"models/{self.model_id}")

    def transcribe(self, audio_np, language="fr", progress_callback=None):
        """
        Transcribe audio and return (words, duration).
        """
        if self.use_albert:
            return self._transcribe_albert(audio_np, language, progress_callback)
        
        if self.model_id == "mock":
            duration = len(audio_np) / 16000
            return [{"start": 0.0, "end": duration, "word": "[MOCK] Transcription de test mélangée (Micro + Système)"}], duration

        if self.model is None:
            self.load()

        duration = len(audio_np) / 16000

        if self.model_id == "whisper":
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
                        all_words.append({"start": w.start, "end": w.end, "word": w.word})
                else:
                    all_words.append({"start": s.start, "end": s.end, "word": s.text.strip()})

            return all_words, info.duration

        else:
            # Vosk
            import vosk  # noqa: PLC0415
            import json
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)

            pcm16 = (audio_np * 32768).astype(np.int16).tobytes()
            rec.AcceptWaveform(pcm16)
            res = json.loads(rec.FinalResult())

            all_words = []
            if "result" in res:
                for w in res["result"]:
                    all_words.append({"start": w["start"], "end": w["end"], "word": w["word"]})

            return all_words, len(audio_np) / 16000

    def _transcribe_albert(self, audio_np, language="fr", progress_callback=None):
        import ffmpeg
        import io
        import requests
        import time

        duration_total = len(audio_np) / 16000

        def _find_best_cut(audio_np, target_sec, search_range=10):
            sr = 16000
            start_idx = max(0, int((target_sec - search_range) * sr))
            end_idx = min(len(audio_np), int((target_sec + search_range) * sr))
            
            chunk_to_search = audio_np[start_idx:end_idx]
            if len(chunk_to_search) < sr * 0.1:
                return target_sec
            
            win_size = int(sr * 0.1) # 100ms
            num_wins = len(chunk_to_search) // win_size
            if num_wins == 0:
                return target_sec
                
            # Calcul de l'energie RMS par fenetre de 100ms
            wins = chunk_to_search[:num_wins*win_size].reshape(num_wins, win_size)
            energies = np.sqrt(np.mean(wins**2, axis=1))
            best_win = np.argmin(energies)
            return (start_idx + best_win * win_size) / sr

        # 1. Découpage en tranches intelligentes (autour de 20 min)
        tranches = []
        t = 0
        while t < duration_total:
            remaining = duration_total - t
            if remaining <= CHUNK_LIMIT_SEC + 30: # +30s de marge finale
                tranches.append((t, remaining))
                break
            
            # Chercher un silence autour de t + CHUNK_LIMIT_SEC sur une fenetre de 3 minutes (+/- 90s)
            cut_point = _find_best_cut(audio_np, t + CHUNK_LIMIT_SEC, search_range=90)
            tranches.append((t, cut_point - t))
            t = cut_point

        all_final_words = []

        for idx, (start_time, duration) in enumerate(tranches):
            if progress_callback:
                # Plage 45% -> 85% du total du pipeline pour la transcription Albert
                prog_base = 45 + int((idx / len(tranches)) * 40)
                progress_callback(f"Albert (Tranche {idx+1}/{len(tranches)})", prog_base)

            # 2. Extraction de la tranche audio en mémoire
            start_sample = int(start_time * 16000)
            end_sample = int((start_time + duration) * 16000)
            chunk_audio = audio_np[start_sample:end_sample]

            # 3. Compression MP3 (48kbps) de la tranche
            import ffmpeg
            from backend.core.audio import float32_to_pcm16
            try:
                pcm16_bytes = float32_to_pcm16(chunk_audio)
                out, _ = (
                    ffmpeg.input("pipe:0", format="s16le", ar=16000, ac=1)
                    .output("pipe:0", format="mp3", acodec="libmp3lame", audio_bitrate="48k")
                    .run(input=pcm16_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
                )
                buffer = io.BytesIO(out)
                mime_type = "audio/mpeg"
                file_extension = "mp3"
            except Exception as e:
                print(f"[!] Compression MP3 échouée sur tranche {idx+1}, repli sur WAV : {e}")
                import soundfile as sf
                buffer = io.BytesIO()
                sf.write(buffer, chunk_audio, 16000, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                mime_type = "audio/wav"
                file_extension = "wav"

            # 4. Appel API Albert avec Retries pour cette tranche
            try:
                headers = {"Authorization": f"Bearer {self.albert_api_key}"}
                size_mb = len(buffer.getvalue()) / (1024 * 1024)
                # Log ultra-clair pour la console serveur
                print(f"[*] [Albert] Envoi Tranche {idx+1}/{len(tranches)} | Durée: {duration/60:.1f} min | Taille: {size_mb:.2f} Mo")
                import sys
                sys.stdout.flush()

                files = {"file": (f"audio.{file_extension}", buffer, mime_type)}
                data = {
                    "model": self.albert_model_id,
                    "language": language,
                    "response_format": "verbose_json",
                }

                max_retries = 3
                last_err = None
                response = None
                for attempt in range(max_retries):
                    try:
                        buffer.seek(0) # IMPORTANT: Remettre au debut pour chaque tentative
                        response = requests.post(
                            f"{self.albert_base_url}/audio/transcriptions",
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=1800,  # 30 minutes
                        )
                        if response.status_code == 200:
                            break
                        else:
                            # Ne retenter que sur des erreurs serveur ou Rate Limit
                            if response.status_code in [429, 500, 502, 503, 504]:
                                wait = 2 ** attempt
                                print(f"[!] Albert API Error {response.status_code} (attempt {attempt+1}/{max_retries}). Retrying in {wait}s...")
                                time.sleep(wait)
                            else:
                                break
                    except (requests.exceptions.RequestException, Exception) as e:
                        last_err = e
                        print(f"[!] Erreur reseau Albert (tentative {attempt+1}): {e}")
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt
                            time.sleep(wait)
                        else:
                            raise last_err

                if not response or response.status_code != 200:
                    status = response.status_code if response else "N/A"
                    txt = response.text if response else str(last_err)
                    raise Exception(f"Échec transcription tranche {idx+1} (Statut {status}): {txt}")

                result = response.json()
                chunk_words = []

                # --- Extraction des mots (verbose_json) ---
                if result.get("words"):
                    for w in result["words"]:
                        chunk_words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
                elif result.get("segments"):
                    for s in result["segments"]:
                        if s.get("words"):
                            for w in s["words"]:
                                chunk_words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
                        else:
                            chunk_words.append({
                                "start": s.get("start", 0.0),
                                "end": s.get("end", duration),
                                "word": s.get("text", "").strip(),
                            })
                elif result.get("text"):
                    chunk_words.append({"start": 0.0, "end": duration, "word": result["text"].strip()})

                # 5. Décalage des timestamps pour cette tranche
                for w in chunk_words:
                    all_final_words.append({
                        "start": w["start"] + start_time,
                        "end": w["end"] + start_time,
                        "word": w["word"]
                    })

            except Exception as e:
                print(f"[!] Erreur critique sur tranche {idx+1} : {e}")
                raise
            finally:
                buffer.close()

        return all_final_words, duration_total
