import torch
import warnings
import numpy as np
import threading
from typing import Optional, Union

# Suppress pkg_resources deprecation warning from webrtcvad
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    import webrtcvad

from silero_vad import load_silero_vad

# Mathematical constant for 16-bit audio normalization
INT16_MAX_ABS_VALUE = 32768.0
SAMPLE_RATE = 16000


class VADManager:
    """
    Combined Silero + WebRTC Voice Activity Detector.

    Two-stage approach inspired by TranscriptionSuite:
    1. WebRTC VAD performs fast initial screening (any single frame triggers)
    2. Silero VAD confirms with higher accuracy (runs in background thread)

    Voice is considered active only when BOTH detectors agree.
    """

    def __init__(
        self,
        silero_sensitivity: float = 0.4,
        webrtc_sensitivity: int = 3,
        silero_use_onnx: bool = True,
    ):
        # --- WebRTC VAD (fast gate) ---
        self.webrtc_vad = webrtcvad.Vad()
        self.webrtc_vad.set_mode(webrtc_sensitivity)
        print(f"[*] WebRTC VAD initialized (sensitivity={webrtc_sensitivity})")

        # --- Silero VAD (accurate confirmation) ---
        self.silero_model = load_silero_vad(onnx=silero_use_onnx)
        self.silero_sensitivity = silero_sensitivity
        print(f"[*] Silero VAD initialized (sensitivity={silero_sensitivity}, onnx={silero_use_onnx})")

        # State tracking (read by the caller between calls)
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self._silero_working = False
        self._lock = threading.Lock()

    def reset_states(self) -> None:
        """Reset all VAD states between sessions."""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        if hasattr(self.silero_model, "reset_states"):
            self.silero_model.reset_states()

    # ------------------------------------------------------------------
    # Stage 1 — WebRTC (fast, works on 10/20/30ms frames)
    # ------------------------------------------------------------------
    def _check_webrtc(self, chunk_pcm: bytes, all_frames_must_be_true: bool = False) -> bool:
        """Quick WebRTC check. Returns True if frame(s) contain speech."""
        frame_len = 320
        num_frames = len(chunk_pcm) // frame_len
        speech_frames = 0

        for i in range(num_frames):
            frame = chunk_pcm[i * frame_len : (i + 1) * frame_len]
            if self.webrtc_vad.is_speech(frame, SAMPLE_RATE):
                speech_frames += 1
                if not all_frames_must_be_true:
                    self.is_webrtc_speech_active = True
                    return True

        if all_frames_must_be_true:
            speech_detected = speech_frames == num_frames
            self.is_webrtc_speech_active = speech_detected
            return speech_detected

        self.is_webrtc_speech_active = False
        return False

    # ------------------------------------------------------------------
    # Stage 2 — Silero (accurate, runs in background thread)
    # ------------------------------------------------------------------
    def _check_silero(self, chunk_pcm: bytes) -> None:
        """Run Silero VAD on the full chunk (called from background thread)."""
        with self._lock:
            self._silero_working = True
            try:
                audio_np = (
                    np.frombuffer(chunk_pcm, dtype=np.int16)
                    .astype(np.float32) / INT16_MAX_ABS_VALUE
                )
                
                chunk_size = 512
                # Get max VAD probability across all 512-sample chunks
                max_vad_prob = 0.0
                for i in range(0, len(audio_np), chunk_size):
                    piece = audio_np[i : i + chunk_size]
                    if len(piece) < chunk_size:
                        piece = np.pad(piece, (0, chunk_size - len(piece)))
                        
                    prob = self.silero_model(
                        torch.from_numpy(piece), SAMPLE_RATE
                    ).item()
                    if prob > max_vad_prob:
                        max_vad_prob = prob

                self.is_silero_speech_active = max_vad_prob > (1 - self.silero_sensitivity)
            finally:
                self._silero_working = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_voice_activity(self, chunk_pcm: bytes) -> None:
        """
        Perform dual-gate voice check (non-blocking).

        Call this for every incoming chunk, then read `is_voice_active()`
        to get the combined result.
        """
        # Fast gate
        self._check_webrtc(chunk_pcm)

        # If WebRTC says speech, launch Silero in background (if not already running)
        if self.is_webrtc_speech_active and not self._silero_working:
            threading.Thread(
                target=self._check_silero,
                args=(chunk_pcm,),
                daemon=True,
            ).start()

    def is_voice_active(self) -> bool:
        """True only when BOTH WebRTC and Silero agree there is speech."""
        return self.is_webrtc_speech_active and self.is_silero_speech_active

    # Legacy synchronous API (kept for simple use cases / tests)
    def is_speech(self, chunk_pcm: bytes) -> bool:
        """Synchronous dual-gate check (blocks on Silero). Used to detect speech onset."""
        if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=False):
            return False
        # Full Silero check synchronously
        with self._lock:
            self._silero_working = True
            try:
                audio_np = (
                    np.frombuffer(chunk_pcm, dtype=np.int16)
                    .astype(np.float32) / INT16_MAX_ABS_VALUE
                )
                
                chunk_size = 512
                # Get max VAD probability across all 512-sample chunks
                max_vad_prob = 0.0
                for i in range(0, len(audio_np), chunk_size):
                    piece = audio_np[i : i + chunk_size]
                    if len(piece) < chunk_size:
                        piece = np.pad(piece, (0, chunk_size - len(piece)))
                        
                    prob = self.silero_model(
                        torch.from_numpy(piece), SAMPLE_RATE
                    ).item()
                    if prob > max_vad_prob:
                        max_vad_prob = prob

                result = max_vad_prob > (1 - self.silero_sensitivity)
                self.is_silero_speech_active = result
                return result
            finally:
                self._silero_working = False

    def check_deactivation(self, chunk_pcm: bytes) -> bool:
        """Strict check for end of speech to aggressively cut off silence.
        Matches TranscriptionSuite behavior where ANY silence frame causes WebRTC to drop."""
        # Use strict WebRTC matching
        if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=True):
            return False
            
        return self.is_speech(chunk_pcm)
