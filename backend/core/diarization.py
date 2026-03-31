import numpy as np
from typing import Any
from backend.config import setup_gpu
import backend.core.speaker_profiles as speaker_profiles
from resemblyzer import preprocess_wav, VoiceEncoder  # type: ignore[import-untyped]

# Ensure NNPACK is disabled for diarization operations
import os
os.environ.setdefault("DISABLE_NNPACK", "1")

class DiarizationEngine:
    def __init__(self, model_id: str = "pyannote/speaker-diarization-3.1", hf_token: str | None = None, use_cpu: bool = False) -> None:
        self.model_id = model_id
        self.hf_token = hf_token
        self.pipeline = None
        self.sample_rate = 16000
        self.use_cpu = use_cpu
        self._fallback_until: float = 0.0
        self.encoder: Any = None  # Resemblyzer encoder for embedding extraction

    def load(self) -> None:
        """
        Attempt to load the heavy Pyannote pipeline and Resemblyzer encoder.
        If the required third‑party libraries are unavailable, fall back to a
        lightweight stub that simply returns empty diarisation results.
        """
        if not self.hf_token:
            print("[!] Warning: No HF_TOKEN provided for Diarization.")
            return

        try:
            # Lazy import of pyannote.audio – may not be installed in the test env.
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]
            import torch

            print(f"[*] Loading Pyannote pipeline: {self.model_id}")
            self.pipeline = Pipeline.from_pretrained(
                self.model_id,
                token=self.hf_token
            )

            if torch.cuda.is_available() and not self.use_cpu:
                if self.pipeline is not None:
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                setup_gpu()
                print("[*] Pyannote loaded on GPU.")
            else:
                print("[*] Pyannote loaded on CPU.")
        except Exception as e:
            # Graceful fallback – keep ``self.pipeline`` as ``None``.
            self.pipeline = None
            print(f"[!] Pyannote could not be loaded ({e}); using fallback diarisation.")

        # Load Resemblyzer encoder (CPU only, lightweight) – optional.
        if self.encoder is None:
            try:
                self.encoder = VoiceEncoder("cpu")
            except Exception:
                self.encoder = None
                print("[!] Resemblyzer encoder could not be loaded; embeddings will be skipped.")

    def diarize(self, audio_float32: np.ndarray, hook: Any = None) -> list[tuple[float, float, str]]:
        if self.pipeline is None:
            # No heavy pipeline – return empty list to keep the pipeline functional.
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
            print(f"[!] Diarisation output error: {type(annotation)}")
            return []

        # First pass: collect raw segments
        raw_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_segments.append((turn.start, turn.end, speaker))

        # Second pass: compute embeddings per segment and attempt speaker ID match
        final_segments = []
        for start, end, speaker in raw_segments:
            # Extract audio slice for the segment
            start_idx = int(start * self.sample_rate)
            end_idx = int(end * self.sample_rate)
            segment_audio = audio_float32[start_idx:end_idx]

            # Compute embedding using Resemblyzer (if encoder is available)
            if self.encoder is not None:
                try:
                    wav = preprocess_wav(segment_audio)
                    emb = self.encoder.embed_utterance(wav)
                    match = speaker_profiles.match_embedding(np.array(emb, dtype=np.float32))
                    if match:
                        _, name = match
                        final_speaker = name
                    else:
                        final_speaker = speaker
                except Exception:
                    # Fallback to original speaker label on any error
                    final_speaker = speaker
            else:
                # No encoder available – keep original speaker label
                final_speaker = speaker

            final_segments.append((start, end, final_speaker))

        return final_segments