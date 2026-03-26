# backend/core/diarization_cpu.py
import numpy as np
import soundfile as sf
import tempfile, os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import backend.core.speaker_profiles as speaker_profiles
from resemblyzer import VoiceEncoder, preprocess_wav

class LightDiarizationEngine:
    def __init__(self):
        self.encoder = None

    def load(self):
        if self.encoder is None:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder("cpu")

    def diarize(self, audio_float32, hook=None):
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_float32)

        # Embeddings sur fenêtres glissantes 1.5s / step 0.5s
        sr, win, step = 16000, int(1.5 * 16000), int(0.5 * 16000)
        times, embeddings = [], []

        for start in range(0, len(wav) - win, step):
            chunk = wav[start:start + win]
            embeddings.append(self.encoder.embed_utterance(chunk))
            times.append(start / sr)

        if len(embeddings) < 2:
            return [(0.0, len(audio_float32) / 16000, "SPEAKER_00")]

        X = np.array(embeddings)
        n = self._estimate_speakers(X)
        labels = AgglomerativeClustering(n_clusters=n).fit_predict(X)

        # ------------------------------------------------------------------
        # Associate clusters with speaker profiles (SQLite)
        # ------------------------------------------------------------------
        # Compute mean embedding per cluster
        cluster_means = {}
        for idx, lbl in enumerate(labels):
            cluster_means.setdefault(lbl, []).append(embeddings[idx])
        cluster_mean_vectors = {lbl: np.mean(vecs, axis=0) for lbl, vecs in cluster_means.items()}

        # Match each cluster mean to a stored profile
        cluster_label_map = {}
        for lbl, mean_vec in cluster_mean_vectors.items():
            match = speaker_profiles.match_embedding(mean_vec)
            if match:
                _, name = match
                cluster_label_map[lbl] = name
            else:
                cluster_label_map[lbl] = f"SPEAKER_{lbl:02d}"

        segments = self._labels_to_segments(times, labels, step / sr)

        # Replace generic speaker labels with matched names where applicable
        segments = [
            (s, e, cluster_label_map.get(int(label.split('_')[-1]), label))
            for (s, e, label) in segments
        ]

        return self._inject_overlaps(audio_float32, segments)

    def _estimate_speakers(self, X, max_spk=6):
        best_n, best_score = 2, -1
        for n in range(2, min(max_spk + 1, len(X))):
            labels = AgglomerativeClustering(n_clusters=n).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, best_n = score, n
        return best_n

    def _labels_to_segments(self, times, labels, step_sec):
        segments = []
        for t, spk in zip(times, labels):
            label = f"SPEAKER_{spk:02d}"
            end = t + step_sec
            if segments and segments[-1][2] == label:
                segments[-1] = (segments[-1][0], end, label)
            else:
                segments.append((t, end, label))
        return segments

    def _inject_overlaps(self, audio, segments, window=1.0, step=0.25):
        sr, win = 16000, int(window * 16000)
        overlaps = []
        for start in range(0, len(audio) - win, int(step * sr)):
            chunk = audio[start:start + win]
            fft = np.abs(np.fft.rfft(chunk))
            if np.sum(fft > np.mean(fft) * 2.5) > len(fft) * 0.15:
                overlaps.append((start / sr, (start + win) / sr))

        result = []
        for s, e, spk in segments:
            is_ov = any(os <= e and oe >= s for os, oe in overlaps)
            result.append((s, e, f"{spk}+OVERLAP" if is_ov else spk))
        return result
