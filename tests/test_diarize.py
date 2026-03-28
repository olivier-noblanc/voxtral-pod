import time
import json
import os
import pytest
from diarize import diarize

# === Configuration ===
AUDIO_FILE = "test_3h.wav"
OUTPUT_JSON = "diarize_results.json"

def test_diarize_benchmark():
    """
    Test de performance (benchmark) pour la diarisation batch.
    S'assure que le RTF est < 1.0 sur un fichier réel.
    """
    if not os.path.exists(AUDIO_FILE):
        pytest.skip(f"Fichier de test {AUDIO_FILE} absent. Saut du benchmark.")

    # === Benchmark ===
    start_time = time.time()
    # La fonction diarize() retourne un objet DiarizeResult
    result = diarize(AUDIO_FILE)
    elapsed = time.time() - start_time

    # Extraction et conversion des segments (objets -> dict) pour compatibilité
    segments = [
        {"start": float(s.start), "end": float(s.end), "speaker": str(s.speaker)}
        for s in result.segments
    ]

    # === Analyse simple ===
    total_audio_time = max(s["end"] for s in segments)
    rtf = elapsed / total_audio_time
    speakers = set(s["speaker"] for s in segments)
    seg_per_min = len(segments) / (total_audio_time / 60)
    avg_segment_duration = sum(s["end"] - s["start"] for s in segments) / len(segments)

    # === Logs ===
    print(f"\n=== DIARIZE BATCH RESULT ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"RTF: {rtf:.3f}")
    print(f"Speakers: {len(speakers)}")
    print(f"Segments/min: {seg_per_min:.1f}")
    print(f"Avg segment duration: {avg_segment_duration:.2f}s")
    print(f"Total segments: {len(segments)}")

    # === Export JSON ===
    with open(OUTPUT_JSON, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"\n[RESULTS SAVED] {OUTPUT_JSON}")

    # === Checks rapides ===
    assert rtf < 1.0, "Temps d'exécution trop long !"
    assert avg_segment_duration > 0.5, "Segments trop courts, vérifier clustering"

if __name__ == "__main__":
    # Permet de lancer le script directement avec 'python tests/test_diarize.py'
    try:
        test_diarize_benchmark()
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")