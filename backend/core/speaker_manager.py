import os
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from backend.core.audio import decode_audio

# Lazy loading of speechbrain
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        logging.info("[*] Loading SpeechBrain ECAPA-TDNN model (CPU)...")
        # Ensure it runs on CPU as requested
        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )
    return _encoder

class SpeakerManager:
    """
    Manages speaker enrollment and identification using SpeechBrain embeddings.
    Strictly CPU-only, offline-ready (after initial model download).
    """
    def __init__(self, db_path: str = "speaker_signatures.json", threshold: float = 0.75):
        self.db_path = db_path
        self.threshold = threshold
        self.signatures: Dict[str, List[float]] = {}
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.signatures = json.load(f)
                logging.info(f"[*] Loaded {len(self.signatures)} speaker signatures from {self.db_path}")
            except Exception as e:
                logging.error(f"[!] Failed to load speaker signatures: {e}")
                self.signatures = {}

    def _save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.signatures, f, indent=2)
            logging.info(f"[*] Saved speaker signatures to {self.db_path}")
        except Exception as e:
            logging.error(f"[!] Failed to save speaker signatures: {e}")

    def get_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        """Extract a 192-dim embedding from audio."""
        import torch
        encoder = get_encoder()
        # Convert to tensor and add batch dim
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        # Extract embedding
        emb = encoder.encode_batch(audio_tensor)
        # Flatten and normalize
        emb = emb.squeeze().detach().numpy()
        # Ensure it's unit length for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

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

    def identify_speaker(self, audio_np: np.ndarray) -> Tuple[str, float]:
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

    def identify_speakers_in_segments(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
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
