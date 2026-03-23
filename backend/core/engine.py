import os
import asyncio
import torch
import numpy as np
from backend.config import setup_warnings, get_vram_gb
from backend.core.audio import decode_audio
from backend.core.diarization import DiarizationEngine
from backend.core.transcription import TranscriptionEngine
from backend.core.merger import assign_speakers_to_words, smooth_micro_turns, build_speaker_segments
from backend.config import setup_warnings

class SotaASR:
    def __init__(self, model_id="whisper", hf_token=None):
        setup_warnings()
        self.albert_api_key = os.getenv("ALBERT_API_KEY")
        
        # 1. Hardware detection
        cuda_available = torch.cuda.is_available()
        vram = get_vram_gb()
        self.no_gpu = not cuda_available
        self.low_vram = (vram < 4.0) if cuda_available else True
        
        if self.no_gpu:
            print("[*] No compatible CUDA device found. Using CPU optimized mode.")
        elif self.low_vram:
            print(f"[*] Low VRAM detected ({vram:.1f}GB). Optimizing for low resources.")

        # 2. Strategy selection
        # If no GPU or low VRAM AND we have Albert key, force Albert transcription
        if self.albert_api_key and (self.low_vram or self.no_gpu or model_id == "albert"):
            print("[*] Fallback: Using Albert API for transcription (No GPU or Low VRAM).")
            self.model_id = "albert"
        else:
            self.model_id = model_id
            
        self.hf_token = hf_token
        
        # Diarization on CPU if no GPU or low VRAM
        self.diarization_engine = DiarizationEngine(hf_token=hf_token, use_cpu=self.low_vram)
        self.transcription_engine = TranscriptionEngine(model_id=self.model_id)
        
        self._loaded = False

    def load(self):
        if self._loaded: return
        self.diarization_engine.load()
        self.transcription_engine.load()
        self._loaded = True

    async def process_file(self, audio_path, progress_callback=None):
        """
        Full pipeline: Decode -> Diarize -> Transcribe -> Merge -> Format
        """
        if not self._loaded:
            self.load()

        # 1. Decode
        if progress_callback: progress_callback("Décodage audio...", 2)
        audio_np = await asyncio.to_thread(decode_audio, audio_path)
        duration = len(audio_np) / 16000

        # 2. Diarize
        if progress_callback: progress_callback("Diarisation...", 5)
        
        def diar_hook(step_name, *args, **kwargs):
            if not progress_callback:
                return
            
            try:
                # Extract raw values
                c_raw = args[0] if len(args) > 0 else kwargs.get('completed', 0)
                t_raw = args[1] if len(args) > 1 else kwargs.get('total')
                
                def to_scalar(v):
                    if v is None: return 0.0
                    if hasattr(v, "item"): return float(v.item())
                    if isinstance(v, (list, np.ndarray)) and len(v) > 0: return float(v[0])
                    return float(v)

                c_val = to_scalar(c_raw)
                t_val = to_scalar(t_raw)

                # Map Pyannote steps to human readable stages
                display_step = step_name
                if "segmentation" in step_name.lower(): display_step = "Détéction voix"
                elif "embedding" in step_name.lower(): display_step = "Identification speakers"
                
                if t_val > 0:
                    # Diarization is about 40% of the total process (5-45%)
                    sub_pct = int((c_val / t_val) * 40)
                    progress_callback(f"Diarisation : {display_step}", 5 + sub_pct)
                else:
                    progress_callback(f"Diarisation : {display_step}...", 5)
            except Exception:
                pass

        diar_segments = await asyncio.to_thread(self.diarization_engine.diarize, audio_np, hook=diar_hook)

        # 3. Transcribe
        if progress_callback: progress_callback("Transcription en cours...", 45)
        
        # Note: TranscriptionEngine returns words directly now
        words, _ = await asyncio.to_thread(self.transcription_engine.transcribe, audio_np, progress_callback=progress_callback)
        if progress_callback: progress_callback("Transcription terminée", 95)

        # 4. Merge (TranscriptionSuite Reference)
        words_with_speakers = assign_speakers_to_words(words, diar_segments)
        words_with_speakers = smooth_micro_turns(words_with_speakers)
        
        segments = build_speaker_segments(words_with_speakers)

        # 5. Format final text
        output_lines = []
        for s in segments:
            output_lines.append(f"[{s['start']:.2f}s -> {s['end']:.2f}s] [{s['speaker']}] {s['text']}")
        
        return "\n".join(output_lines)
