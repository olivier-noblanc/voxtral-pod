import warnings
import numpy as np
import asyncio
import concurrent.futures
import os

# Suppress noisy UserWarning from webrtcvad about pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

# Attempt to configure PyTorch/NNPACK before any heavy usage
try:
    import torch
    if hasattr(torch, "backends") and hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.enabled = False
except ImportError:
    pass

# Lazy import helper for webrtcvad
def _lazy_get_webrtcvad():
    """Attempt to import webrtcvad; if unavailable, provide a fallback stub."""
    import webrtcvad
    return webrtcvad


# Lazy import of silero_vad
def _lazy_load_silero_vad(onnx: bool = True):
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=onnx)


# Mathematical constant for 16-bit audio normalization
INT16_MAX_ABS_VALUE = 32768.0
SAMPLE_RATE = 16000
_SILERO_CHUNK_SIZE = 512

# Global thread pool for VAD to limit resources
_VAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 4) + 4)
)


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
        aggressive: bool = False,
        silero_use_onnx: bool = True,
    ):
        # --- WebRTC VAD (fast gate) ---
        webrtcvad = _lazy_get_webrtcvad()
        self.webrtc_vad = webrtcvad.Vad()
        self.webrtc_vad.set_mode(webrtc_sensitivity)
        print(f"[*] WebRTC VAD initialized (sensitivity={webrtc_sensitivity})")

        # --- Silero VAD (accurate confirmation) ---
        self.silero_model = _lazy_load_silero_vad(onnx=silero_use_onnx)
        self.silero_sensitivity = silero_sensitivity
        print(
            f"[*] Silero VAD initialized (sensitivity={silero_sensitivity}, onnx={silero_use_onnx})"
        )

        # Store aggressive mode flag
        self.aggressive = aggressive

        # State tracking (read by the caller between calls)
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self._lock = asyncio.Lock()
        self._silero_semaphore = asyncio.Semaphore(1)

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
    # Stage 2 — Silero (accurate)
    # ------------------------------------------------------------------
    def _run_silero(self, chunk_pcm: bytes) -> float:
        """
        Calcule la probabilité maximale de parole sur le chunk via Silero.
        Factorisation : utilisée à la fois par `_check_silero` (thread) et `is_speech` (sync).
        """
        audio_np = (
            np.frombuffer(chunk_pcm, dtype=np.int16)
            .astype(np.float32)
            / INT16_MAX_ABS_VALUE
        )
        max_vad_prob = 0.0
        # Process audio in fixed-size chunks to avoid padding issues
        for i in range(0, len(audio_np), _SILERO_CHUNK_SIZE):
            piece = audio_np[i : i + _SILERO_CHUNK_SIZE]
            if len(piece) < _SILERO_CHUNK_SIZE:
                piece = np.pad(piece, (0, _SILERO_CHUNK_SIZE - len(piece)))
            # Lazy import torch only when needed
            import torch
            # Extra safety: ensure NNPACK remains disabled in executor threads
            if hasattr(torch, "backends") and hasattr(torch.backends, "nnpack") and torch.backends.nnpack.enabled:
                torch.backends.nnpack.enabled = False

            prob = self.silero_model(torch.from_numpy(piece), SAMPLE_RATE).item()
            if prob > max_vad_prob:
                max_vad_prob = prob
        return max_vad_prob

    def _run_silero_sync(self, chunk_pcm: bytes) -> bool:
        """
        Version synchrone interne pour mise à jour de l'état Silero.
        Utilisée par run_in_executor.
        """
        max_vad_prob = self._run_silero(chunk_pcm)
        result = max_vad_prob > (1 - self.silero_sensitivity)
        self.is_silero_speech_active = result
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def is_speech(self, chunk_pcm: bytes) -> bool:
        """Asynchronous dual-gate check. Uses a global thread pool for Silero."""
        # Stage 1: WebRTC (very fast)
        if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=False):
            return False

        # Stage 2: Silero (heavier)
        # We use a semaphore to ensure only one Silero calculation happens at a time per session
        async with self._silero_semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _VAD_POOL,
                self._run_silero_sync,
                chunk_pcm
            )

    def is_speech_sync(self, chunk_pcm: bytes) -> bool:
        """Legacy synchronous dual-gate check. Primarily for tests or non-async contexts."""
        if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=False):
            return False
        return self._run_silero_sync(chunk_pcm)

    async def check_deactivation(self, chunk_pcm: bytes) -> bool:
        """
        Check for end‑of‑speech asynchronously.
        """
        if self.aggressive:
            # Aggressive: any silence frame from WebRTC stops speech quickly.
            if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=False):
                return False
            # Still run Silero to confirm speech continuity when WebRTC still sees speech.
            return await self.is_speech(chunk_pcm)
        else:
            # Legacy behaviour: require *all* frames to be speech before considering deactivation.
            if not self._check_webrtc(chunk_pcm, all_frames_must_be_true=True):
                return False
            return await self.is_speech(chunk_pcm)