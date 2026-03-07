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
    def __init__(self, silero_sensitivity: float = 0.4):
        # WebRTC VAD (Rapide, pour un premier filtrage) - OBLIGATOIRE
        import webrtcvad
        self.webrtc_vad = webrtcvad.Vad(3) # Niveau d'agression max (3)
        print("[*] WebRTC VAD chargé.")

        # Silero VAD (Lent mais ultra précis, pour confirmer)
        from silero_vad import load_silero_vad
        
        # Load Silero VAD model
        self.silero_model = load_silero_vad(onnx=True)
        self.silero_sensitivity = silero_sensitivity
        
        self.is_webrtc_speech = False
        self.is_silero_speech = False
        self._lock = threading.Lock()

    def is_speech(self, audio_bytes: bytes) -> bool:
        """Détecte si le segment audio contient de la parole (Double Validation)"""
        # 1. Premier passage : WebRTC (si dispo)
        if self.webrtc_vad:
            is_speech_webrtc = False
            # WebRTC travaille sur des frames de 30ms (480 samples @ 16kHz)
            frame_size = 480 * 2
            for i in range(0, len(audio_bytes), frame_size):
                frame = audio_bytes[i:i+frame_size]
                if len(frame) < frame_size: break
                if self.webrtc_vad.is_speech(frame, SAMPLE_RATE):
                    is_speech_webrtc = True
                    break
            
            if not is_speech_webrtc:
                return False
        
        # 2. Deuxième passage : Silero Check (précision)
        with self._lock:
            # Silero VAD v5+ n'accepte QUE des chunks de 512 samples (32ms) à 16kHz
            # Si on a plus, on prend les 512 derniers pour la détection "live"
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / INT16_MAX_ABS_VALUE
            
            if len(audio_np) > 512:
                audio_np = audio_np[-512:]
            elif len(audio_np) < 512:
                # Trop court pour Silero, on complète avec du silence (padding)
                audio_np = np.pad(audio_np, (0, 512 - len(audio_np)))
                
            vad_prob = self.silero_model(torch.from_numpy(audio_np), SAMPLE_RATE).item()
            return vad_prob > (1 - self.silero_sensitivity)
