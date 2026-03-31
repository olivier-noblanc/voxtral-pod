from typing import Any
from backend.core.audio import decode_audio

import os
import json
import logging
import threading
import numpy as np

# Ensure NNPACK is disabled for speaker management operations
os.environ.setdefault("DISABLE_NNPACK", "1")

# Lazy loading of speechbrain
_encoder: Any = None
_encoder_lock = threading.Lock()
_encoder_tried = False

# Silence speechbrain fetching logs which are very noisy
logging.getLogger("speechbrain.utils.fetching").setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.parameter_transfer").setLevel(logging.WARNING)

def get_encoder() -> Any:
    global _encoder, _encoder_tried
    if _encoder is not None:
        return _encoder

    with _encoder_lock:
        if _encoder is not None:
            return _encoder

        if _encoder_tried:
            # Already tried and failed, don't retry and spam logs/network
            return None

        _encoder_tried = True
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore[import-untyped]
            logging.info("[*] Loading SpeechBrain ECAPA-TDNN model (CPU singleton)...")
            # Ensure it runs on CPU as requested
            _encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
        except Exception as e:
            logging.error(f"[!] Failed to load SpeechBrain model: {e}")
            # Keep _encoder as None, but _encoder_tried is True to prevent retries
            pass

    return _encoder

class SpeakerManager:
    """
    Manages speaker enrollment and identification using SpeechBrain embeddings.
    Strictly CPU-only, offline-ready (after initial model download).
    """
    def __init__(self, db_path: str = "speaker_signatures.json", threshold: float = 0.75) -> None:
        self.db_path = db_path
        self.threshold = threshold
        self.signatures: dict[str, list[float]] = {}
        self._load_db()

    def _load_db(self) -> None:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.signatures = json.load(f)
                logging.info(f"[*] Loaded {len(self.signatures)} speaker signatures from {self.db_path}")
            except Exception as e:
                logging.error(f"[!] Failed to load speaker signatures: {e}")
                self.signatures = {}

    def _save_db(self) -> None:
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.signatures, f, indent=2)
            logging.info(f"[*] Saved speaker signatures to {self.db_path}")
        except Exception as e:
            logging.error(f"[!] Failed to save speaker signatures: {e}")

    def get_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract a 192-dim embedding from audio."""
        return self.get_embeddings_batch([audio_np])[0]

    def get_embeddings_batch(self, audio_chunks: list[np.ndarray]) -> list[np.ndarray]:
        """Extract embeddings for a list of audio chunks using true batching (one forward pass)."""
        import torch
        encoder = get_encoder()
        if encoder is None:
            raise RuntimeError("Failed to load speaker encoder model.")

        if not audio_chunks:
            return []

        # 1. Prepare batch with padding
        # SpeechBrain expects [batch, time]
        max_len = max(len(c) for c in audio_chunks)
        batch_size = len(audio_chunks)
        
        # Create padded tensor
        wavs = torch.zeros(batch_size, max_len)
        wav_lens = torch.zeros(batch_size)
        
        for i, chunk in enumerate(audio_chunks):
            clent = len(chunk)
            wavs[i, :clent] = torch.from_numpy(chunk.copy())
            wav_lens[i] = clent / max_len
            
        # 2. Extract embeddings in one go
        # SpeechBrain handling padding internally via wav_lens
        try:
            with torch.no_grad():
                # encode_batch returns [batch, 1, 192] for ECAPA-TDNN
                emb = encoder.encode_batch(wavs, wav_lens)
                emb = emb.squeeze(1).detach().cpu().numpy()
        except Exception as e:
            logging.error(f"[!] Batch inference failed: {e}. Falling back to sequential.")
            # Fallback if batch is too large for RAM or other issues
            return [self.get_embedding(c) for c in audio_chunks]

        # 3. Normalize and format
        results = []
        for i in range(batch_size):
            e = emb[i]
            norm = np.linalg.norm(e)
            if norm > 1e-6:
                e = e / norm
            results.append(e)
            
        return results

    def enroll_speaker(self, name: str, audio_path: str) -> bool:
        """Enroll a speaker by extracting an embedding from the provided audio file."""
        try:
            audio_np = decode_audio(audio_path)
            emb = self.get_embedding(audio_np)
            self.signatures[name] = emb.tolist()
            self._save_db()
            return True
        except Exception as e:
            logging.error(f"[!] Enrollment failed for {name}: {e}")
            return False

    def identify_speaker(self, audio_np: np.ndarray) -> tuple[str, float]:
        """
        Identify the speaker from an audio fragment.
        Returns (name, score). Returns ("Unknown", 0.0) if no match.
        """
        if not self.signatures:
            return "Unknown", 0.0

        try:
            current_emb = self.get_embedding(audio_np)
            best_name = "Unknown"
            best_score = 0.0

            for name, sig_list in self.signatures.items():
                sig = np.array(sig_list)
                # Since embeddings are normalized, cosine similarity is just the dot product
                score = float(np.dot(current_emb, sig))
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= self.threshold:
                return best_name, best_score
            return "Unknown", best_score
        except Exception as e:
            logging.error(f"[!] Identification failed: {e}")
            return "Unknown", 0.0

    def identify_speakers_in_segments(self, audio_path: str, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Takes a list of segments [{start, end}] and identifies the speaker for each.
        """
        full_audio = decode_audio(audio_path)
        sample_rate = 16000
        
        results = []
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            chunk = full_audio[start_sample:end_sample]
            
            if len(chunk) < 1600: # Ignore segments < 0.1s
                name, score = "Too Short", 0.0
            else:
                name, score = self.identify_speaker(chunk)
            
            results.append({
                "start": seg['start'],
                "end": seg['end'],
                "speaker": name,
                "confidence": score
            })
        return results
