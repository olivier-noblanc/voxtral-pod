"""
Moteur de transcription pour Voxtral Pod.
Gère Whisper (local), Vosk (local) et l'API Albert (distant).
"""
import os
import io
import sys
from typing import Any
import time
import requests
import numpy as np
import soundfile as sf
import subprocess
from backend.core.postprocess import _ensure_ffmpeg
from backend.core.albert_rate_limiter import albert_rate_limiter

# Propriété utilitaire : nom des modèles qui utilisent l'API Albert
_ALBERT_MODEL_IDS = frozenset({"albert"})

# Limit de segmentation pour l'API Albert (300s = 50 min)
# On reste sous 20 Mo grâce à la compression MP3 48k (50 min ≈ 17,2 Mo)
# Limite de taille du payload (en Mo) – on vise < 20 Mo
MAX_PAYLOAD_MB = 19
# Chunk limit disabled – aucune contrainte de durée fixe ; la segmentation s’appuie sur la taille du payload
# Nous limitons la durée maximale d’un chunk à 40 minutes (2400 s). Le bitrate est ajusté à 64 kbit/s afin que le MP3 reste < 20 Mo même pour la durée maximale.
CHUNK_LIMIT_SEC = 2400

# Additional fix for KaldiRecognizer cleanup issue
import warnings
import os
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*KaldiRecognizer")

# Ensure NNPACK is disabled for transcription operations
os.environ.setdefault("DISABLE_NNPACK", "1")

class TranscriptionEngine:
    """
    Orchestrateur de transcription multi-moteur.
    """
    def __init__(self, model_id="whisper", device=None):
        self.model_id = model_id

        # Automatic device detection (lazy import de torch)
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Any = None
        self.albert_api_key = os.getenv("ALBERT_API_KEY")
        self.albert_base_url = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
        self.albert_model_id = os.getenv("ALBERT_MODEL_ID", "openai/whisper-large-v3")

    @property
    def use_albert(self) -> bool:
        """True si l'on doit passer par l'API Albert pour la transcription."""
        return self.model_id in _ALBERT_MODEL_IDS or bool(self.albert_api_key)

    def load(self):
        """Charge les modèles en mémoire (Whisper ou Vosk)."""
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

    def transcribe(self, audio_np, language="fr", progress_callback=None, is_partial: bool = False):
        """
        Transcribe audio and return (words, duration).
        """
        # Tâche 1 : Optimisation des transcriptions partielles (CPU pur)
        # Si c'est une transcription partielle, forcer l'utilisation du modèle local
        if is_partial and self.model_id != "mock":
            # Forcer l'utilisation du modèle local pour les transcriptions partielles
            use_local_model = True
        else:
            use_local_model = not self.use_albert or self.model_id == "mock"

        # NOTE : les transcriptions partielles sont toujours réalisées en local (CPU)
        # Elles ne passent jamais par l'API Albert, même si le fallback est actif.
        # Le basculement vers le modèle CPU ne dépend donc pas du rate‑limiter.
        
        # Tâche 2 : Bascule de secours (Fallback) sur limite de requêtes
        if self.use_albert and not use_local_model and albert_rate_limiter.should_use_cpu_fallback_mode():
            # Tâche 2 : Bascule silencieuse sur le modèle local (CPU)
            print(f"[*] Basculage sur modèle local (fallback CPU) en raison du rate limit")
            use_local_model = True
            # Charger le modèle local si nécessaire
            if self.model is None:
                self.load()
        
        if use_local_model:
            # Utiliser le modèle local (Whisper ou Vosk)
            if self.model_id == "mock":
                duration = len(audio_np) / 16000
                return [{"start": 0.0, "end": duration, "word": "[MOCK] Transcription de test mélangée (Micro + Système)"}], duration

            if self.model is None:
                self.load()

            duration = len(audio_np) / 16000

            if self.model_id == "whisper":
                segments, info = self.model.transcribe(  # type: ignore
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
                            all_words.append({"start": w.start, "end": w.end, "word": w.word, "speaker": "UNKNOWN"})
                    else:
                        all_words.append({"start": s.start, "end": s.end, "word": s.text.strip(), "speaker": "UNKNOWN"})

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
                        all_words.append({"start": w["start"], "end": w["end"], "word": w["word"], "speaker": "UNKNOWN"})

                return all_words, len(audio_np) / 16000
        else:
            # Utiliser l'API Albert
            # Pour Albert, on passe is_partial comme argument supplémentaire si le modèle le supporte
            try:
                return self._transcribe_albert(audio_np, language, progress_callback, is_partial=is_partial)
            except TypeError as e:
                # Si l'argument is_partial n'est pas supporté, on appelle sans cet argument
                if "unexpected keyword argument 'is_partial'" in str(e):
                    return self._transcribe_albert(audio_np, language, progress_callback)
                raise

    def _find_best_cut(self, audio_np, target_sec, current_t=0):
        """
        Trouve le meilleur point de coupe (silence) près de target_sec.
        On recherche dans une fenêtre réduite à la fin du segment (180s)
        afin de maximiser la durée de la tranche.
        """
        sr = 16000
        # Fenêtre de recherche : les 3 dernières minutes du segment cible
        search_range_sec = 180
        start_sec = max(current_t, target_sec - search_range_sec)
        end_sec = min(len(audio_np) / sr, target_sec)
        
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)

        segment = audio_np[start_idx:end_idx]
        if len(segment) == 0:
            return target_sec

        # Analyse d'énergie par fenêtres de 200ms
        frame_len = int(sr * 0.2)
        num_frames = len(segment) // frame_len
        if num_frames == 0:
            return target_sec

        frames = segment[:num_frames * frame_len].reshape(num_frames, frame_len)
        energies = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

        # Seuil d'énergie de silence
        SILENCE_THRESHOLD = 1e-3

        # On cherche le DERNIER silence dans cette fenêtre de recherche
        silent_indices = np.where(energies < SILENCE_THRESHOLD)[0]
        if silent_indices.size > 0:
            last_silent = silent_indices[-1]
            cut_sec = start_sec + (last_silent * frame_len) / sr
            # Si le silence est très proche de la fin ou du début, on ajuste
            if cut_sec >= target_sec - 0.5 or cut_sec <= start_sec + 0.5:
                return target_sec
            return cut_sec

        # Fallback : cadre avec l'énergie la plus faible
        min_idx = np.argmin(energies)
        cut_sec = start_sec + (min_idx * frame_len) / sr
        if cut_sec >= target_sec - 0.5 or cut_sec <= start_sec + 0.5:
            return target_sec
        return cut_sec

    def _transcribe_albert(self, audio_np, language="fr", progress_callback=None, is_partial: bool = False):
        """
        Découpe l'audio en tranches et les envoie à l'API Albert.
        """
        duration_total = len(audio_np) / 16000

        # Vérifier si on doit utiliser le mode mock en raison du rate limit
        if albert_rate_limiter.is_in_mock_mode():
            print(f"[*] Utilisation du mode mock pour Albert (quota dépassé)")
            # Retourner une transcription mock avec un message d'erreur
            return [{"start": 0.0, "end": duration_total, "word": "[MODE MOCK] Quota API dépassé. Utilisation du mode de secours."}], duration_total

        tranches = []
        t = 0
        while t < duration_total:
            remaining = duration_total - t
            if remaining <= CHUNK_LIMIT_SEC + 60:  # +60s de marge finale
                tranches.append((t, remaining))
                break

            cut_point = self._find_best_cut(audio_np, t + CHUNK_LIMIT_SEC, current_t=int(t))
            tranches.append((t, cut_point - t))
            t = cut_point

        all_final_words = []

        for idx, (start_time, duration) in enumerate(tranches):
            if progress_callback:
                prog_base = 45 + int((idx / len(tranches)) * 40)
                progress_callback(f"Albert ({idx+1}/{len(tranches)})", prog_base)

            chunk_audio = audio_np[int(start_time*16000) : int((start_time+duration)*16000)]
            
            # 1. Compression (fallback to MP3) – utilise ffmpeg via ffmpeg‑python
            import ffmpeg
            _ensure_ffmpeg()
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, chunk_audio, 16000, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)

            # Conversion MP3 64k
            out, err = (
                ffmpeg
                .input('pipe:0', format='wav')  # type: ignore
                .output('pipe:1', format='mp3', audio_bitrate='64k', vbr='on')
                .run(input=wav_buffer.read(), capture_stdout=True, capture_stderr=True)
            )
            buffer = io.BytesIO(out)
            mime_type = "audio/mpeg"
            file_ext = "mp3"

            # 2. Diagnostics détaillés
            raw_bytes = buffer.getvalue()
            size_mb = len(raw_bytes) / (1024*1024)
            print(f"\n--- [DEBUG ALBERT] TRANCHE {idx+1}/{len(tranches)} ---")
            print(f"[*] Durée segment: {duration:.2f} s")
            print(f"[*] Taille payload: {len(raw_bytes)} octets ({size_mb:.2f} Mo)")
            print(f"[*] Format: {file_ext} | MIME: {mime_type}")
            
            headers = {"Authorization": f"Bearer {self.albert_api_key}"}
            files = {"file": (f"audio.{file_ext}", buffer, mime_type)}
            data = {"model": self.albert_model_id, "language": language, "response_format": "verbose_json"}
            
            # 3. Requête avec Retry
            last_err = None
            response = None
            for attempt in range(3):
                start_req = time.time()
                try:
                    # Vérification du rate limit avant l'envoi
                    if not albert_rate_limiter.can_make_request():
                        print(f"[*] Rate limit actif, attente avant requête...")
                        # Attendre jusqu'à l'intervalle requis
                        time.sleep(albert_rate_limiter.min_interval_seconds)
                    
                    buffer.seek(0)
                    print(f"[*] Tentative {attempt+1}/3 - Envoi en cours...")
                    response = requests.post(f"{self.albert_base_url}/audio/transcriptions", headers=headers, files=files, data=data, timeout=1800)
                    print(f"[*] Réponse reçue en {time.time() - start_req:.2f}s (Statut: {response.status_code})")
                    
                    # Enregistrer la requête avant de traiter la réponse
                    albert_rate_limiter.record_request()
                    # Traiter la réponse du rate limiter
                    albert_rate_limiter.handle_response(response.status_code)
                    
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                        print(f"[!] Rate limit (429). Attente {retry_after}s avant retry...")
                        time.sleep(retry_after)
                        response = None  # ← reset pour forcer la détection d'échec
                    elif response.status_code in [500, 502, 503, 504]:
                        print(f"[!] Erreur serveur ({response.status_code}). Retry dans {2**attempt}s...")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[!] Erreur API ({response.status_code}): {response.text}")
                        break  # erreur non récupérable, on sort
                        
                except Exception as e:
                    last_err = e
                    print(f"[!] Exception réseau tranche {idx+1}: {type(e).__name__} - {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)

            if response is None or response.status_code != 200:
                status = response.status_code if response else "N/A"
                detail = response.text[:200] if response else str(last_err)
                raise Exception(f"Échec Tranche {idx+1} (Statut {status}): {detail}")

            # 4. Traitement des résultats
            result = response.json()
            chunk_words = []
            if result.get("words"):
                chunk_words = [
                    {"start": w["start"], "end": w["end"], "word": w["word"], "speaker": "UNKNOWN"}
                    for w in result["words"]
                ]
                print(f"[*] Tranche {idx+1}: {len(chunk_words)} mots récupérés (format: words).")
            elif result.get("segments"):
                segments = result["segments"]
                for s in segments:
                    if s.get("words"):
                        for w in s["words"]: 
                            chunk_words.append({"start": w["start"], "end": w["end"], "word": w["word"], "speaker": "UNKNOWN"})
                    else:
                        chunk_words.append({
                            "start": s.get("start", 0.0), 
                            "end": s.get("end", duration), 
                            "word": s.get("text", "").strip(),
                            "speaker": "UNKNOWN"
                        })
                print(f"[*] Tranche {idx+1}: {len(segments)} segments ({len(chunk_words)} mots) récupérés (format: segments).")
            elif result.get("text"):
                chunk_words.append({"start": 0.0, "end": duration, "word": result["text"].strip(), "speaker": "UNKNOWN"})
                print(f"[*] Tranche {idx+1}: Texte brut récupéré ({len(result['text'])} caractères).")

            # 5. Décalage des timestamps
            if chunk_words:
                print(f"[*] Tranche {idx+1}: Recalage temporel de {len(chunk_words)} éléments (+{start_time:.2f}s)...")
            for w in chunk_words:
                all_final_words.append({
                    "start": w["start"] + start_time, 
                    "end": w["end"] + start_time, 
                    "word": w["word"],
                    "speaker": w.get("speaker", "UNKNOWN")
                })
            
            buffer.close()

        return all_final_words, duration_total
