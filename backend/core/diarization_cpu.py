# backend/core/diarization_cpu.py
import numpy as np
import soundfile as sf
import tempfile, os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import backend.core.speaker_profiles as speaker_profiles
from resemblyzer import VoiceEncoder, preprocess_wav

class LightDiarizationEngine:
    """
    Minimal stub for CPU diarization – returns empty segments.
    """
    def __init__(self):
        self.encoder = None

    def load(self):
        """Placeholder load – does nothing."""
        pass

    def diarize(self, audio_float32, hook=None):
        """Return an empty list of speaker segments."""
        return []