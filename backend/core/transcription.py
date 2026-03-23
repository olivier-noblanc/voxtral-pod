import torch
import numpy as np
import faster_whisper
import vosk
import json
import requests
import tempfile
import os
import io

class TranscriptionEngine:
    def __init__(self, model_id="whisper", device=None):
        self.model_id = model_id
        
        # Automatic device detection
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = None
        self.albert_api_key = os.getenv("ALBERT_API_KEY")
        self.albert_base_url = "https://albert.api.etalab.gouv.fr/v1"
        self.albert_model_id = os.getenv("ALBERT_MODEL_ID", "openai/whisper-large-v3")

    def load(self):
        if self.model_id == "albert" or (self.model_id == "whisper" and self.albert_api_key):
            print("[*] Using Albert API for transcription...")
            # No local model to load for Albert
            return

        if self.model_id == "whisper":
            print(f"[*] Loading Faster-Whisper (large-v3) on {self.device}...")
            # Selection of compute_type based on device
            if self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
                
            self.model = faster_whisper.WhisperModel(
                "large-v3", 
                device=self.device, 
                compute_type=compute_type
            )
        else:
            print(f"[*] Loading Vosk model: {self.model_id}...")
            self.model = vosk.Model(f"models/{self.model_id}")

    def transcribe(self, audio_np, language="fr", progress_callback=None):
        """
        Transcribe audio and return words with timestamps.
        """
        if self.model_id == "albert" or (self.model_id == "whisper" and self.albert_api_key):
            return self._transcribe_albert(audio_np, language, progress_callback)

        if self.model is None:
            self.load()

        duration = len(audio_np) / 16000

        if self.model_id == "whisper":
            segments, info = self.model.transcribe(
                audio_np, 
                beam_size=5, 
                language=language, 
                task="transcribe",
                word_timestamps=True
            )
            
            all_words = []
            for s in segments:
                # Progress update (Whisper is about 50% of the total process: 45-95%)
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
            # Vosk implementation
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)
            
            pcm16 = (audio_np * 32768).astype(np.int16).tobytes()
            rec.AcceptWaveform(pcm16)
            res = json.loads(rec.FinalResult())
            
            all_words = []
            if "result" in res:
                for w in res["result"]:
                    all_words.append({
                        "start": w["start"], 
                        "end": w["end"], 
                        "word": w["word"]
                    })
            
            return all_words, len(audio_np)/16000

    def _transcribe_albert(self, audio_np, language="fr", progress_callback=None):
        if progress_callback: progress_callback("Préparation audio pour Albert...", 46)
        
        buffer = io.BytesIO()
        import scipy.io.wavfile as wavfile
        wavfile.write(buffer, 16000, (audio_np * 32767).astype(np.int16))
        buffer.seek(0)
        
        try:
            if progress_callback: progress_callback("Appel API Albert...", 50)
            
            headers = {"Authorization": f"Bearer {self.albert_api_key}"}
            files   = {'file': ('audio.wav', buffer, 'audio/wav')}
            data    = {
                'model':           self.albert_model_id,   # openai/whisper-large-v3
                'language':        language,
                'response_format': 'verbose_json'          # ← timestamps si dispo
            }
            
            response = requests.post(
                f"{self.albert_base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[!] Albert API Error: {response.text}")
                raise Exception(f"Albert API error: {response.status_code}")
    
            result   = response.json()
            duration = len(audio_np) / 16000
            all_words = []
    
            # verbose_json → mots avec timestamps
            if result.get('words'):
                for w in result['words']:
                    all_words.append({
                        "start": w['start'],
                        "end":   w['end'],
                        "word":  w['word']
                    })
            # verbose_json → segments avec timestamps
            elif result.get('segments'):
                for s in result['segments']:
                    if s.get('words'):
                        for w in s['words']:
                            all_words.append({
                                "start": w['start'],
                                "end":   w['end'],
                                "word":  w['word']
                            })
                    else:
                        # Segment sans mots → fallback timestamps segment
                        all_words.append({
                            "start": s.get('start', 0.0),
                            "end":   s.get('end', duration),
                            "word":  s.get('text', '').strip()
                        })
            # json / text → pas de timestamps
            elif result.get('text'):
                print("[!] Albert: pas de timestamps → segment unique")
                all_words.append({
                    "start": 0.0,
                    "end":   duration,
                    "word":  result['text'].strip()
                })
            else:
                print(f"[!] Albert: réponse inattendue → {str(result)[:200]}")
                return [], 0
    
            return all_words, duration
    
        except Exception as e:
            print(f"[!] Albert API Inference error: {e}")
            return [], 0
        finally:
            buffer.close()
