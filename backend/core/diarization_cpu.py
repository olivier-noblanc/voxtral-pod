# backend/core/diarization_cpu.py
from simple_diarizer.diarizer import Diarizer
import numpy as np, soundfile as sf, tempfile, os

class LightDiarizationEngine:
    def __init__(self):
        self.diarizer = Diarizer(embed_model='ecapa', cluster_method='sc')

    def load(self):
        pass  # lazy load dans simple-diarizer

    def diarize(self, audio_float32, hook=None):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_float32, 16000)
            wav_path = f.name
        try:
            segments = self.diarizer.diarize(wav_path, num_speakers=None)
            result = [(s['start'], s['end'], s['label']) for s in segments]
            return self._inject_overlaps(audio_float32, result)
        finally:
            os.unlink(wav_path)

    def _inject_overlaps(self, audio, segments, window=1.0, step=0.25):
        sr = 16000
        win = int(window * sr)
        overlaps = []
        for start in range(0, len(audio) - win, int(step * sr)):
            chunk = audio[start:start + win]
            fft = np.abs(np.fft.rfft(chunk))
            if np.sum(fft > np.mean(fft) * 2.5) > len(fft) * 0.15:
                overlaps.append((start / sr, (start + win) / sr))

        result = []
        for s, e, spk in segments:
            is_ov = any(os <= e and oe >= s for os, oe in overlaps)
            result.append((s, e, f"{spk}+OVERLAP" if is_ov else spk))
        return result
