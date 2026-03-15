import torch
import numpy as np
import faster_whisper
import vosk
import json

class TranscriptionEngine:
    def __init__(self, model_id="whisper", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None

    def load(self):
        if self.model_id == "whisper":
            print("[*] Loading Faster-Whisper (large-v3)...")
            compute_type = "float16" if torch.cuda.is_available() else "int8"
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
