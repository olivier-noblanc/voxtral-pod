import asyncio
import numpy as np
from fastapi import WebSocket
from backend.core.vad import VADManager, SAMPLE_RATE

# Diarization batch interval (seconds of audio accumulated before running Pyannote)
DIAR_BATCH_INTERVAL_SEC = 300  # 5 minutes

class LiveSession:
    """Isolated per-connection live transcription with VAD."""

    def __init__(self, engine, websocket: WebSocket, client_id: str):
        self.engine = engine
        self.websocket = websocket
        self.client_id = client_id

        self.audio_queue = asyncio.Queue()
        self.pre_speech_buffer = bytearray()
        self.sentence_buffer = bytearray()
        self.full_session_audio = []   # ALL audio (speech + silence) for final pass

        self.is_speaking = False
        self.silence_chunks_count = 0

        # Sentence tracking
        self._sentence_index = 0
        self._sentences = []  
        self._total_bytes_received = 0

        # VAD constants from engine or hardcoded
        self.pre_recording_size = int(SAMPLE_RATE * 2 * 1.0) # 1s
        self.min_segment_bytes = int(SAMPLE_RATE * 2 * 0.5) # 0.5s
        self.silence_chunks_threshold = 9 # ~1.5s

        self.vad = VADManager(silero_sensitivity=0.4)

    async def process_audio_queue(self):
        """Worker: dequeue chunks, run VAD, trigger inference."""
        print(f"[*] [{self.client_id}] Live Audio processor started.")
        self.vad.reset_states()
        
        partial_interval_bytes = int(SAMPLE_RATE * 2 * 2.0)
        last_partial_bytes = 0

        while True:
            audio_bytes = await self.audio_queue.get()
            if audio_bytes is None: break

            self.full_session_audio.append(audio_bytes)
            self._total_bytes_received += len(audio_bytes)

            # Update pre-record buffer
            self.pre_speech_buffer.extend(audio_bytes)
            if len(self.pre_speech_buffer) > self.pre_recording_size:
                self.pre_speech_buffer = self.pre_speech_buffer[-self.pre_recording_size:]

            # VAD logic
            if self.is_speaking:
                has_speech = self.vad.check_deactivation(audio_bytes)
            else:
                has_speech = self.vad.is_speech(audio_bytes)

            if has_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.sentence_buffer = bytearray(self.pre_speech_buffer)
                    last_partial_bytes = len(self.sentence_buffer)
                else:
                    self.sentence_buffer.extend(audio_bytes)
                self.silence_chunks_count = 0
                
                # Partial transcription
                if len(self.sentence_buffer) - last_partial_bytes >= partial_interval_bytes:
                    last_partial_bytes = len(self.sentence_buffer)
                    await self._transcribe_segment(bytes(self.sentence_buffer), final=False)
            else:
                if self.is_speaking:
                    self.sentence_buffer.extend(audio_bytes)
                    self.silence_chunks_count += 1
                    if self.silence_chunks_count >= self.silence_chunks_threshold:
                        self.is_speaking = False
                        self.silence_chunks_count = 0
                        pcm_data = bytes(self.sentence_buffer)
                        self.sentence_buffer = bytearray()
                        if len(pcm_data) >= self.min_segment_bytes:
                            await self._transcribe_segment(pcm_data, final=True)

    async def _transcribe_segment(self, pcm_data, final=True):
        """Invoke engine transcription and send via WS."""
        try:
            # We need a small conversion utility or just use the engine one
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Simple transcription for live (not necessarily word-level diarization yet)
            # In live mode, we usually want text quickly.
            # For now, we reuse the engine's transcription logic
            words, _ = await asyncio.to_thread(self.engine.transcription_engine.transcribe, audio_np)
            text = " ".join(w["word"] for w in words).strip()
            
            if text:
                await self.websocket.send_json({
                    "type": "sentence",
                    "text": text if final else f"{text} ...",
                    "speaker": "Speaker", # Placeholder for now, batch diarization will group later
                    "final": final
                })
        except Exception as e:
            # We catch all exceptions here to prevent the live session worker from crashing
            # but we log it for debugging.
            print(f"[!] [{self.client_id}] Live Inference error: {e}")
