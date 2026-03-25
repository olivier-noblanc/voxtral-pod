import numpy as np
from backend.config import setup_gpu

class DiarizationEngine:
    def __init__(self, model_id="pyannote/speaker-diarization-3.1", hf_token=None, use_cpu=False):
        self.model_id = model_id
        self.hf_token = hf_token
        self.pipeline = None
        self.sample_rate = 16000
        self.use_cpu = use_cpu

    def load(self):
        if not self.hf_token:
            print("[!] Warning: No HF_TOKEN provided for Diarization.")
            return

        # Lazy import of pyannote.audio
        from pyannote.audio import Pipeline
        import torch

        print(f"[*] Loading Pyannote pipeline: {self.model_id}")
        self.pipeline = Pipeline.from_pretrained(
            self.model_id, 
            token=self.hf_token
        )
        
        if torch.cuda.is_available() and not self.use_cpu:
            self.pipeline = self.pipeline.to(torch.device("cuda"))
            setup_gpu()
            print("[*] Pyannote loaded on GPU.")
        else:
            print("[*] Pyannote loaded on CPU.")

    def diarize(self, audio_float32, hook=None):
        if self.pipeline is None:
            return []

        # Lazy import of torch
        import torch

        # Ensure TF32 is enabled
        setup_gpu()

        waveform = torch.from_numpy(audio_float32[None, :])
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": self.sample_rate},
            hook=hook
        )

        # Robust extraction (Pyannote 4.x)
        annotation = getattr(diarization, "speaker_diarization", 
                            getattr(diarization, "annotation", diarization))

        if not hasattr(annotation, "itertracks"):
            print(f"[!] Diarization output error: {type(annotation)}")
            return []

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        return segments
