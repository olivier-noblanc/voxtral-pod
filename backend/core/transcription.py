import os
import io
import sys
import time
import requests
import numpy as np

# Propriété utilitaire : nom des modèles qui utilisent l'API Albert
_ALBERT_MODEL_IDS = frozenset({"albert"})

# Limite de segmentation pour l'API Albert (3000s = 50 min)
# On reste sous 20Mo grace a la compression MP3 48k (50 min ~= 17.2Mo)
CHUNK_LIMIT_SEC = 3000

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
        self.albert_base_url = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
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
            import faster_whisper
            self.model = faster_whisper.WhisperModel(
                "large-v3",
                device=self.device,
                compute_type=compute_type,
            )
        else:
            import vosk
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
            import json
            import vosk
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
        duration_total = len(audio_np) / 16000

        def _find_best_cut(audio_np, target_sec, search_range=90):
            sr = 16000
            # Fenetre de recherche centree sur target_sec, mais apres t
            search_start_sec = max(t + CHUNK_LIMIT_SEC * 0.5, target_sec - search_range)
            start_idx = int(search_start_sec * sr)
            end_idx = min(len(audio_np), int((target_sec + search_range) * sr))
            
            chunk_to_search = audio_np[start_idx:end_idx]
            if len(chunk_to_search) < sr * 0.5:
                return target_sec
            
            win_size = int(sr * 0.2) # 200ms
            num_wins = len(chunk_to_search) // win_size
            if num_wins == 0:
                return target_sec
            
            wins = chunk_to_search[:num_wins*win_size].reshape(num_wins, win_size)
            energies = np.sqrt(np.mean(wins**2, axis=1))
            
            # Penalité de distance pour preferer le point le plus proche de target_sec
            win_times = search_start_sec + (np.arange(num_wins) * win_size) / sr
            # On divise par un grand nombre pour que l'energie reste prioritaire
            dist_penalty = np.abs(win_times - target_sec) * 1e-7
            
            best_win = np.argmin(energies + dist_penalty)
            return win_times[best_win]

        tranches = []
        t = 0
        while t < duration_total:
            remaining = duration_total - t
            if remaining <= CHUNK_LIMIT_SEC + 60: # +60s de marge finale
                tranches.append((t, remaining))
                break
            
            cut_point = _find_best_cut(audio_np, t + CHUNK_LIMIT_SEC)
            tranches.append((t, cut_point - t))
            t = cut_point

        all_final_words = []

        for idx, (start_time, duration) in enumerate(tranches):
            if progress_callback:
                prog_base = 45 + int((idx / len(tranches)) * 40)
                progress_callback(f"Albert ({idx+1}/{len(tranches)})", prog_base)

            chunk_audio = audio_np[int(start_time*16000) : int((start_time+duration)*16000)]
            
            import ffmpeg
            
            # 1. Compression MP3
            try:
                from backend.core.audio import float32_to_pcm16
                pcm16_bytes = float32_to_pcm16(chunk_audio)
                out, err = (
                    ffmpeg.input("pipe:0", format="s16le", ar=16000, ac=1)
                    .output("pipe:0", format="mp3", acodec="libmp3lame", audio_bitrate="48k")
                    .run(input=pcm16_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
                )
                buffer = io.BytesIO(out)
                mime_type = "audio/mpeg"
                file_ext = "mp3"
            except Exception as e:
                print(f"\n[!] ERREUR CRITIQUE COMPRESSION MP3 (Tranche {idx+1})")
                print(f"[!] Erreur: {e}")
                if hasattr(e, "stderr") and e.stderr:
                    print(f"[!] FFmpeg STDERR:\n{e.stderr.decode()}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                
                import soundfile as sf
                buffer = io.BytesIO()
                sf.write(buffer, chunk_audio, 16000, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                mime_type = "audio/wav"
                file_ext = "wav"

            # 2. Diagnostics détaillés
            raw_bytes = buffer.getvalue()
            size_mb = len(raw_bytes) / (1024*1024)
            print(f"\n--- [DEBUG ALBERT] TRANCHE {idx+1}/{len(tranches)} ---")
            print(f"[*] Durée segment: {duration/60:.2f} min")
            print(f"[*] Taille payload: {len(raw_bytes)} octets ({size_mb:.2f} Mo)")
            print(f"[*] Format: {file_ext} | MIME: {mime_type}")
            
            safe_key = (self.albert_api_key[:6] + "..." + self.albert_api_key[-4:]) if self.albert_api_key else "None"
            headers = {"Authorization": f"Bearer {self.albert_api_key}"}
            print(f"[*] Headers: {{'Authorization': 'Bearer {safe_key}', ...}}")

            files = {"file": (f"audio.{file_ext}", buffer, mime_type)}
            data = {"model": self.albert_model_id, "language": language, "response_format": "verbose_json"}
            
            # 3. Requête avec Retry et Timings
            last_err = None
            response = None
            for attempt in range(3):
                try:
                    buffer.seek(0)
                    start_req = time.time()
                    print(f"[*] Tentative {attempt+1}/3 - Envoi en cours...")
                    response = requests.post(f"{self.albert_base_url}/audio/transcriptions", headers=headers, files=files, data=data, timeout=1800)
                    end_req = time.time()
                    print(f"[*] Réponse reçue en {end_req - start_req:.2f}s (Statut: {response.status_code})")
                    if response.status_code == 200:
                        break
                    else:
                        print(f"[!] Erreur API ({response.status_code}): {response.text}")
                        if response.status_code in [429, 500, 502, 503, 504]:
                            time.sleep(2**attempt)
                        else:
                            break
                except Exception as e:
                    end_req = time.time()
                    last_err = e
                    print(f"[!] Exception réseau tranche {idx+1} après {end_req - start_req:.2f}s: {type(e).__name__} - {e}")
                    if attempt < 2:
                        time.sleep(2**attempt)
                    else:
                        raise last_err

            if not response or response.status_code != 200:
                raise Exception(f"Échec Tranche {idx+1} (Statut {response.status_code if response else 'N/A'}): {response.text if response else str(last_err)}")

            # 4. Traitement des mots
            result = response.json()
            chunk_words = []
            if result.get("words"):
                for w in result["words"]: chunk_words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
            elif result.get("segments"):
                for s in result["segments"]:
                    if s.get("words"):
                        for w in s["words"]: chunk_words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
                    else:
                        chunk_words.append({"start": s.get("start", 0.0), "end": s.get("end", duration), "word": s.get("text", "").strip()})
            elif result.get("text"):
                chunk_words.append({"start": 0.0, "end": duration, "word": result["text"].strip()})

            # 5. Décalage des timestamps
            for w in chunk_words:
                all_final_words.append({"start": w["start"] + start_time, "end": w["end"] + start_time, "word": w["word"]})
            
            buffer.close()

        return all_final_words, duration_total
