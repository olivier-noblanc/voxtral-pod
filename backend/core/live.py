import asyncio
# import struct  # Removed: manual WAV header construction; using soundfile instead
import numpy as np
import os
import datetime
import shutil
from fastapi import WebSocket
from backend.core.vad import VADManager, SAMPLE_RATE
from backend.config import TRANSCRIPTIONS_DIR


class LiveSession:
    """Isolated per-connection live transcription with VAD."""

    def __init__(self, engine, websocket: WebSocket, client_id: str, partial_albert: bool = False):
        self.engine = engine
        self.websocket = websocket
        self.client_id = client_id
        self.partial_albert = partial_albert

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
        self.pre_recording_size = int(SAMPLE_RATE * 2 * 1.0)   # 1s de pré-buffer
        self.min_segment_bytes = int(SAMPLE_RATE * 2 * 0.5)    # segment minimum : 0.5s
        self.silence_chunks_threshold = 9                        # ~1.5s de silence

        self.vad = VADManager(silero_sensitivity=0.4)

    async def process_audio_queue(self):
        """Worker: dequeue chunks, run VAD, trigger inference."""
        mode_str = " (Albert API)" if self.engine.model_id == "albert" else ""
        partial_str = " [Partials ON]" if self.partial_albert else " [Partials OFF]"
        print(f"[*] [{self.client_id}] Live Audio processor started{mode_str}{partial_str}.")
        self.vad.reset_states()

        partial_interval_bytes = int(SAMPLE_RATE * 2 * 2.0)
        last_partial_bytes = 0

        while True:
            audio_bytes = await self.audio_queue.get()
            if audio_bytes is None:
                break

            self.full_session_audio.append(audio_bytes)
            self._total_bytes_received += len(audio_bytes)

            # Update pre-record buffer
            self.pre_speech_buffer.extend(audio_bytes)
            if len(self.pre_speech_buffer) > self.pre_recording_size:
                self.pre_speech_buffer = self.pre_speech_buffer[-self.pre_recording_size:]

            # VAD logic – protect against VAD failures
            try:
                if self.is_speaking:
                    has_speech = self.vad.check_deactivation(audio_bytes)
                else:
                    has_speech = self.vad.is_speech(audio_bytes)
            except Exception as e:
                # En cas d’erreur du VAD, on considère qu’il y a de la parole
                print(f"[!] [{self.client_id}] VAD error: {e} – assuming speech")
                has_speech = True

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
                    is_albert = self.engine.model_id == "albert"
                    if not is_albert or self.partial_albert:
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

    async def save_audio_file(self) -> str | None:
        """Sauvegarde les données audio de la session dans un fichier WAV valide,
        puis génère une transcription complète propre du fichier audio."""
        if not self.full_session_audio:
            return None

        # ---------- Enregistrement du fichier WAV ----------
        audio_dir = os.path.join(TRANSCRIPTIONS_DIR, "live_audio")
        os.makedirs(audio_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = f"live_{self.client_id}_{timestamp}.wav"
        wav_path = os.path.join(audio_dir, wav_filename)

        pcm_bytes = b"".join(self.full_session_audio)
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Lazy import of soundfile
        import soundfile as sf
        sf.write(wav_path, audio_np, SAMPLE_RATE, subtype='PCM_16')

        print(f"[*] [{self.client_id}] Audio saved: {wav_filename} ({len(pcm_bytes) / 1024:.1f} KB)")

        # ---------- Transcription complète du fichier audio ----------
        # audio_np déjà calculé ci-dessus — pas besoin de le recalculer

        # Utiliser le moteur de transcription pour obtenir le texte complet
        words, _ = await asyncio.to_thread(self.engine.transcription_engine.transcribe, audio_np)
        full_text = " ".join(w["word"] for w in words).strip()

        # Enregistrer la transcription propre dans un fichier .txt à côté du wav
        txt_filename = wav_filename.replace(".wav", ".txt")
        txt_path = os.path.join(audio_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(full_text)

        print(f"[*] [{self.client_id}] Full transcription saved: {txt_filename}")

        # ---------- Copier la transcription dans le répertoire client ----------
        client_dir = os.path.join(TRANSCRIPTIONS_DIR, self.client_id)
        os.makedirs(client_dir, exist_ok=True)
        shutil.copy(txt_path, client_dir)

        return wav_filename

    async def _transcribe_segment(self, pcm_data: bytes, final: bool = True):
        """Invoke engine transcription and send via WS."""
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            words, _ = await asyncio.to_thread(self.engine.transcription_engine.transcribe, audio_np)
            text = " ".join(w["word"] for w in words).strip()

            if text:
                await self.websocket.send_json({
                    "type": "sentence",
                    "text": text if final else f"{text} ...",
                    "speaker": "Speaker",  # Placeholder — la diarisation par lot viendra plus tard
                    "final": final,
                })
        except Exception as e:
            # On capture toutes les exceptions pour ne pas faire crasher le worker live
            print(f"[!] [{self.client_id}] Live Inference error: {e}")
