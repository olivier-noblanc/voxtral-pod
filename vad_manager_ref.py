"""Reference VAD manager for testing purposes.

Provides a simple implementation using webrtcvad and silero_vad.
"""

import warnings
import torch
import webrtcvad
import numpy as np
from silero_vad import load_silero_vad
import threading

# Suppress noisy UserWarning from webrtcvad about pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

def _lazy_get_webrtcvad():
    """Import webrtcvad and return the module."""
    return webrtcvad

def _lazy_load_silero_vad(onnx: bool = True):
    """Load the silero VAD model."""
    return load_silero_vad(onnx=onnx)

class VADManagerRef:
    """Simple VAD manager using webrtcvad and silero_vad."""

    def __init__(
        self,
        webrtc_sensitivity: int = 3,
        silero_sensitivity: float = 0.4,
        aggressive: bool = False,
        silero_use_onnx: bool = True,
    ):
        self.webrtc_vad = _lazy_get_webrtcvad().Vad()
        self.webrtc_vad.set_mode(webrtc_sensitivity)
        self.silero_model = _lazy_load_silero_vad(onnx=silero_use_onnx)
        self.silero_sensitivity = silero_sensitivity
        self.aggressive = aggressive

        self.is_webrtc_active = False
        self.is_silero_active = False
        self._working = False
        self._lock = threading.Lock()

    def reset(self):
        """Reset detection flags."""
        self.is_webrtc_active = False
        self.is_silero_active = False

    def _check_webrtc(self, chunk: bytes, all_frames: bool = False) -> bool:
        """Check voice activity using webrtcvad."""
        frame_len = 320
        num_frames = len(chunk) // frame_len
        speech_frames = 0

        for i in range(num_frames):
            frame = chunk[i * frame_len : (i + 1) * frame_len]
            if self.webrtc_vad.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames:
                    self.is_webrtc_active = True
                    return True

        if all_frames:
            active = speech_frames == num_frames
            self.is_webrtc_active = active
            return active

        self.is_webrtc_active = False
        return False

    def _run_silero(self, chunk: bytes) -> float:
        """Run silero VAD on the audio chunk and return max probability."""
        audio_np = (
            np.frombuffer(chunk, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )
        max_prob = 0.0
        for i in range(0, len(audio_np), 512):
            piece = audio_np[i : i + 512]
            if len(piece) < 512:
                piece = np.pad(piece, (0, 512 - len(piece)))
            prob = self.silero_model(torch.from_numpy(piece), 16000).item()
            if prob > max_prob:
                max_prob = prob
        return max_prob

    def _check_silero(self, chunk: bytes) -> None:
        """Check silero VAD and update state."""
        with self._lock:
            self._working = True
            try:
                prob = self._run_silero(chunk)
                self.is_silero_active = prob > (1 - self.silero_sensitivity)
            finally:
                self._working = False

    def check_voice_activity(self, chunk: bytes) -> None:
        """Public method to start voice activity detection."""
        self._check_webrtc(chunk)

        if self.is_webrtc_active and not self._working:
            threading.Thread(
                target=self._check_silero,
                args=(chunk,),
                daemon=True,
            ).start()

    def is_voice_active(self) -> bool:
        """Return True when both detectors agree."""
        return self.is_webrtc_active and self.is_silero_active