import numpy as np
import time
from vad_manager import VADManager

CHUNK_SIZE = 2560 # 160ms chunks
SAMPLE_RATE = 16000
vad = VADManager(silero_sensitivity=0.4, webrtc_sensitivity=3)

# Generate 5 seconds of silence
silence = np.zeros(int(SAMPLE_RATE * 5), dtype=np.int16).tobytes()

# Generate 2 seconds of noise/sine wave (mock speech)
t = np.linspace(0, 2, int(SAMPLE_RATE * 2))
speech = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).tobytes()

print("Feeding Silence (2 sec)...")
for i in range(0, len(silence)//2, CHUNK_SIZE * 2):
    chunk = silence[i:i + CHUNK_SIZE * 2]
    if len(chunk) < CHUNK_SIZE * 2: continue
    is_speech = vad.is_speech(chunk)
    print(f"Silence chunk: speech={is_speech}")

print("\nFeeding Speech (2 sec)...")
for i in range(0, len(speech), CHUNK_SIZE * 2):
    chunk = speech[i:i + CHUNK_SIZE * 2]
    if len(chunk) < CHUNK_SIZE * 2: continue
    is_speech = vad.is_speech(chunk)
    print(f"Speech chunk: speech={is_speech}")
    
print("\nFeeding Silence again (2 sec)...")
for i in range(0, len(silence)//2, CHUNK_SIZE * 2):
    chunk = silence[i:i + CHUNK_SIZE * 2]
    if len(chunk) < CHUNK_SIZE * 2: continue
    is_speech = vad.is_speech(chunk)
    print(f"Silence chunk: speech={is_speech}")

