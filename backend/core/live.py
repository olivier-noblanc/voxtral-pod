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

    def __init__(self, engine, websocket: WebSocket, client_id: str, partial_albert: bool = False, selected_audio_device_id: str | None = None):
        self.engine = engine
        self.websocket = websocket
        self.client_id = client_id
        self.partial_albert = False  # Désactiver les partials par défaut pour éviter les appels fréquents à l'API

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
        self.silence_chunks_threshold = 5                        # ~0.8s de silence

        self.vad = VADManager(silero_sensitivity=0.4)
        self.selected_audio_device_id = selected_audio_device_id
        self._lock = asyncio.Lock()

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
                # Finalisation du dernier segment en cours à la fermeture de la session
                if self.is_speaking and self.sentence_buffer:
                    pcm_data = bytes(self.sentence_buffer)
                    # Note: on finalise même si c'est court, pour éviter un résidu "..."
                    await self._transcribe_segment(pcm_data, final=True)
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
                    has_speech = await self.vad.check_deactivation(audio_bytes)
                else:
                    has_speech = await self.vad.is_speech(audio_bytes)
            except Exception as e:
                # En cas d’erreur du VAD, on considère qu’il y a de la parole
                print(f"[!] [{self.client_id}] VAD error: {e} – assuming speech")
                has_speech = True

            if has_speech:
                if not self.is_speaking:
                    print(f"[*] [{self.client_id}] Voice detected (VAD START)")
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
                        print(f"[*] [{self.client_id}] Silence detected (VAD END)")
                        self.is_speaking = False
                        self.silence_chunks_count = 0
                        pcm_data = bytes(self.sentence_buffer)
                        self.sentence_buffer = bytearray()
                        if len(pcm_data) >= self.min_segment_bytes:
                            await self._transcribe_segment(pcm_data, final=True)
                            # On vide le pre_speech_buffer pour éviter que la fin de cette phrase 
                            # ne soit reprise comme contexte (et donc répétée) au début de la suivante.
                            self.pre_speech_buffer = bytearray()

    # Retourne ``(None, None)`` lorsqu'aucune donnée audio n'est disponible.
    # Le typage indique explicitement que les deux valeurs peuvent être ``None``.
    async def save_wav_only(self) -> tuple[None | str, None | str]:
        """Sauvegarde les données audio de la session dans un fichier WAV valide."""
        if not self.full_session_audio:
            return None, None

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
        abs_wav_path = os.path.abspath(wav_path)
        print(f"[*] [{self.client_id}] Audio saved: {wav_filename} ({len(pcm_bytes) / 1024:.1f} KB) at {abs_wav_path}")

        return wav_path, timestamp

    async def _transcribe_segment(self, pcm_data: bytes, final: bool = True):
        """Invoke engine transcription and send via WS."""
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Tâche 1 : Optimisation des transcriptions partielles (CPU pur)
            # Passer l'état final pour déterminer si c'est une transcription partielle
            try:
                words, _ = await asyncio.to_thread(self.engine.transcription_engine.transcribe, audio_np, is_partial=not final)
            except TypeError as e:
                # Si l'argument is_partial n'est pas supporté, on appelle sans cet argument
                if "unexpected keyword argument 'is_partial'" in str(e):
                    words, _ = await asyncio.to_thread(self.engine.transcription_engine.transcribe, audio_np)
                else:
                    raise
            text = " ".join(w["word"] for w in words).strip()

            if text or final:
                async with self._lock:
                    # Nettoyage des éventuels points de suspension (souvent au début ou fin avec Albert/Whisper)
                    display_text = text
                    if final:
                        display_text = text.strip(". ").strip()

                    await self.websocket.send_json({
                        "type": "sentence",
                        "text": display_text,
                        "speaker": "Speaker",  # Placeholder — la diarisation par lot viendra plus tard
                        "final": final,
                        "sentence_index": self._sentence_index,
                        "total_bytes_received": self._total_bytes_received,
                    })

                    # Stocker la phrase finalisée et incrémenter l'index SEULEMENT APRÈS l'envoi
                    if final:
                        self._sentences.append(text)
                        self._sentence_index += 1
                # Debug print to console
                tag = "[FINAL]" if final else "[PARTIAL]"
                print(f"[*] [{self.client_id}] {tag}: {text}")
        except Exception as e:
            # On capture toutes les exceptions pour ne pas faire crasher le worker live
            print(f"[!] [{self.client_id}] Live Inference error: {e}")
            # Notifier le client si c'est un rate limit
            if "429" in str(e) or "rate limit" in str(e).lower():
                try:
                    await self.websocket.send_json({
                        "type": "error",
                        "message": "Quota API dépassé (1000 req/jour). Réessayez demain.",
                        "final": True,
                        "sentence_index": self._sentence_index,
                        "total_bytes_received": self._total_bytes_received,
                    })
                except Exception:
                    pass
