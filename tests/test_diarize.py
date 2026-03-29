import time
import json
import os
import tempfile
import soundfile as sf
import pytest
from diarize import diarize

# === Configuration ===
AUDIO_FILE = "test_3h.wav"
OUTPUT_JSON = "diarize_results.json"
MAX_BENCH_DURATION = 120.0  # seconds (2 minutes for a fast benchmark)

def get_heaviest_audio_file(search_dir="."):
    """Finds the largest audio file in the specified directory recursively."""
    extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
    heaviest_file = None
    max_size = -1

    for root, _, files in os.walk(search_dir):
        # Skip some boring directories
        if any(x in root for x in ["venv", "node_modules", ".git", "__pycache__"]):
            continue
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                path = os.path.join(root, file)
                try:
                    size = os.path.getsize(path)
                    if size > max_size:
                        max_size = size
                        heaviest_file = path
                except OSError:
                    continue
    return heaviest_file

def test_diarize_benchmark():
    """
    Test de performance (benchmark) pour la diarisation batch.
    S'assure que le RTF est < 1.0 sur un fichier réel.
    """
    audio_to_use = AUDIO_FILE
    if not os.path.exists(audio_to_use):
        print(f"[*] Default file {AUDIO_FILE} missing, searching for fallback...")
        audio_to_use = get_heaviest_audio_file()
        
    if not audio_to_use or not os.path.exists(audio_to_use):
        pytest.skip("Aucun fichier audio trouvé pour le benchmark. Saut du test.")

    print(f"[*] Using audio file for benchmark: {audio_to_use} ({os.path.getsize(audio_to_use)/1024/1024:.1f} MB)")

    # === Truncation for faster benchmark ===
    temp_audio_file = None
    try:
        info = sf.info(audio_to_use)
        if info.duration > MAX_BENCH_DURATION:
            print(f"[*] Truncating audio from {info.duration:.1f}s to {MAX_BENCH_DURATION}s for a faster benchmark.")
            # Create a temporary WAV file with the truncated audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_file = tmp.name
            
            # Read only the first MAX_BENCH_DURATION seconds
            # sf.read(..., frames=...) is efficient
            frames_to_read = int(MAX_BENCH_DURATION * info.samplerate)
            data, sr = sf.read(audio_to_use, frames=frames_to_read)
            sf.write(temp_audio_file, data, sr)
            audio_to_use = temp_audio_file

        # === Benchmark ===
        start_time = time.time()
        # La fonction diarize() retourne un objet DiarizeResult
        result = diarize(audio_to_use)
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
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            print(f"[*] Cleaning up temporary file: {temp_audio_file}")
            os.remove(temp_audio_file)

if __name__ == "__main__":
    # Permet de lancer le script directement avec 'python tests/test_diarize.py'
    try:
        test_diarize_benchmark()
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")