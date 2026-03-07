import os
import torch
import warnings
import numpy as np
import threading
from typing import Optional, Union

# Mathematical constant for 16-bit audio normalization
INT16_MAX_ABS_VALUE = 32768.0
SAMPLE_RATE = 16000

class VADManager:
    """
    Gestionnaire de VAD combinant WebRTC (rapide) et Silero (précis).
    Inspiré de TranscriptionSuite.
    """
    def __init__(self, silero_sensitivity: float = 0.4, webrtc_sensitivity: int = 3):
        import webrtcvad
        from silero_vad import load_silero_vad
        
        self.webrtc_vad = webrtcvad.Vad()
        self.webrtc_vad.set_mode(webrtc_sensitivity)
        
        # Load Silero VAD model
        self.silero_model = load_silero_vad(onnx=True)
        self.silero_sensitivity = silero_sensitivity
        
        self.is_webrtc_speech = False
        self.is_silero_speech = False
        self._lock = threading.Lock()

    def is_speech(self, chunk_pcm: bytes) -> bool:
        """
        Vérifie si le chunk contient de la parole en utilisant les deux VAD.
        La parole est confirmée si les DEUX sont d'accord (ou si Silero confirme après WebRTC).
        """
        # 1. WebRTC Check (très rapide)
        # WebRTC nécessite des frames de 10, 20 ou 30ms. 
        # Pour 16kHz, 10ms = 160 samples (320 bytes).
        frame_len = 320 
        num_frames = len(chunk_pcm) // frame_len
        webrtc_active = False
        for i in range(num_frames):
            frame = chunk_pcm[i*frame_len : (i+1)*frame_len]
            if self.webrtc_vad.is_speech(frame, SAMPLE_RATE):
                webrtc_active = True
                break
        
        if not webrtc_active:
            return False

        # 2. Silero Check (précision)
        with self._lock:
            audio_float = np.frombuffer(chunk_pcm, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
            vad_prob = self.silero_model(torch.from_numpy(audio_float), SAMPLE_RATE).item()
            return vad_prob > (1 - self.silero_sensitivity)
