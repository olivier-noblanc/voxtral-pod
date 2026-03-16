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
        import scipy.io.wavfile as wavfile
        
        if progress_callback: progress_callback("Préparation audio pour Albert...", 46)
        
        # Albert needs a file object. We'll use a temporary WAV in memory/disk.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wavfile.write(temp_path, 16000, (audio_np * 32767).astype(np.int16))
        
        try:
            if progress_callback: progress_callback("Appel API Albert...", 50)
            
            headers = {
                "Authorization": f"Bearer {self.albert_api_key}"
            }
            
            # According to docs, we need 'file', 'model', and 'response_format'
            # Assuming large-v3 is available under some ID, but docs say "automatic-speech-recognition"
            # We'll try to find the model name or use a default if possible.
            # For now, let's use "fluently-asr" or similar if known, or just a placeholder.
            # Actually, Albert API usually provides a specific model string.
            
            with open(temp_path, 'rb') as audio_file:
                files = {
                    'file': audio_file
                }
                data = {
                    'model': 'automatic-speech-recognition', # Placeholder ID from docs example
                    'language': language,
                    'response_format': 'verbose_json' # We need word timestamps if possible
                }
                
                response = requests.post(
                    f"{self.albert_base_url}/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
            
            if response.status_code != 200:
                print(f"[!] Albert API Error: {response.text}")
                raise Exception(f"Albert API error: {response.status_code}")

            result = response.json()
            all_words = []
            
            # Albert verbose_json might look like OpenAI's format
            if 'words' in result:
                for w in result['words']:
                    all_words.append({"start": w['start'], "end": w['end'], "word": w['word']})
            elif 'segments' in result:
                for s in result['segments']:
                    if 'words' in s:
                        for w in s['words']:
                            all_words.append({"start": w['start'], "end": w['end'], "word": w['word']})
                    else:
                        # Fallback to segment text if no words
                        all_words.append({"start": s['start'], "end": s['end'], "word": s['text'].strip()})
            else:
                # Final fallback
                all_words.append({"start": 0, "end": len(audio_np)/16000, "word": result.get('text', '')})

            return all_words, len(audio_np)/16000

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
