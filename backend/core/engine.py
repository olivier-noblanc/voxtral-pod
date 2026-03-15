import os
import asyncio
import numpy as np
from backend.core.audio import decode_audio
from backend.core.diarization import DiarizationEngine
from backend.core.transcription import TranscriptionEngine
from backend.core.merger import assign_speakers_to_words, smooth_micro_turns, build_speaker_segments
from backend.config import setup_warnings

class SotaASR:
    def __init__(self, model_id="whisper", hf_token=None):
        setup_warnings()
        self.model_id = model_id
        self.hf_token = hf_token
        
        self.diarization_engine = DiarizationEngine(hf_token=hf_token)
        self.transcription_engine = TranscriptionEngine(model_id=model_id)
        
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
        audio_np = decode_audio(audio_path)
        duration = len(audio_np) / 16000

        # 2. Diarize
        if progress_callback: progress_callback("Diarisation...", 5)
        
        def diar_hook(step_name, *args, **kwargs):
            if not progress_callback:
                return
            
            # Extract completed and total (robustly)
            completed = args[0] if len(args) > 0 else kwargs.get('completed', 0)
            if completed is None: completed = 0
            
            total = args[1] if len(args) > 1 else kwargs.get('total')

            if completed is not None and total and total > 0:
                sub_pct = int((completed / total) * 40)
                progress_callback(f"Diarisation ({step_name})...", 5 + sub_pct)

        diar_segments = self.diarization_engine.diarize(audio_np, hook=diar_hook)

        # 3. Transcribe
        if progress_callback: progress_callback("Transcription en cours...", 45)
        
        # Note: TranscriptionEngine returns words directly now
        words, _ = self.transcription_engine.transcribe(audio_np)
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
