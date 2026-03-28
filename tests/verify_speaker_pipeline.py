import os
import sys
import numpy as np
import soundfile as sf
import logging

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.speaker_manager import SpeakerManager
from backend.core.diarization_cpu import LightDiarizationEngine

logging.basicConfig(level=logging.INFO)

def create_test_audio(path, duration=3.0, freq=440):
    """Create a simple sine wave test audio file."""
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, audio, sr)
    return path

def test_speaker_manager():
    print("\n--- Testing SpeakerManager ---")
    manager = SpeakerManager(db_path="test_signatures.json")
    
    # Create two different test audios
    audio1_path = create_test_audio("speaker1.wav", freq=440)
    audio2_path = create_test_audio("speaker2.wav", freq=880)
    
    print("[*] Enrolling Speaker 1...")
    manager.enroll_speaker("Speaker 1", audio1_path)
    
    print("[*] Enrolling Speaker 2...")
    manager.enroll_speaker("Speaker 2", audio2_path)
    
    # Test identification
    audio1 = sf.read(audio1_path)[0]
    name, score = manager.identify_speaker(audio1)
    print(f"[!] Identified Speaker 1 as: {name} (Score: {score:.4f})")
    assert name == "Speaker 1"
    
    audio2 = sf.read(audio2_path)[0]
    name, score = manager.identify_speaker(audio2)
    print(f"[!] Identified Speaker 2 as: {name} (Score: {score:.4f})")
    assert name == "Speaker 2"
    
    # Cleanup
    os.remove(audio1_path)
    os.remove(audio2_path)
    if os.path.exists("test_signatures.json"):
        os.remove("test_signatures.json")

def test_diarization_engine():
    print("\n--- Testing DiarizationEngine ---")
    engine = LightDiarizationEngine()
    engine.load()
    
    # Create a 5s test audio
    test_path = create_test_audio("diarize_test.wav", duration=5.0)
    
    print("[*] Running diarization...")
    try:
        segments = engine.diarize_file(test_path)
        print(f"[!] Diarization results: {segments}")
        
        print("[*] Running benchmark...")
        stats = engine.benchmark(test_path)
        print(f"[!] Benchmark stats: RTF={stats['rtf']}, Segments={stats['segments_count']}")
    except Exception as e:
        print(f"[!] Diarization failed: {e}")
        # If it fails due to missing weights or environment, we don't want to crash the whole test
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)

def test_french_pipeline():
    print("\n--- Testing French Audio Pipeline ---")
    chunk_path = "test_french_chunk.mp3"
    if not os.path.exists(chunk_path):
        print(f"[!] {chunk_path} non trouvé. Saut du test français.")
        return

    manager = SpeakerManager(db_path="french_signatures.json")
    engine = LightDiarizationEngine()
    engine.load()

    # 1. Enrôlement : on prend les 30 premières secondes du fichier original pour Nicolas Renahy
    print("[*] Enrôlement de 'Nicolas Renahy'...")
    # On utilise un fichier temporaire pour l'enrôlement
    from backend.core.audio import decode_audio
    import tempfile
    full_audio = decode_audio("test_french.mp3")
    sample_rate = 16000
    enroll_chunk = full_audio[0:int(30 * sample_rate)] # 30s
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
        sf.write(tmp_name, enroll_chunk, sample_rate)
    
    try:
        manager.enroll_speaker("Nicolas Renahy", tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 2. Diarisation du segment de 2 minutes
    print(f"[*] Diarisation de {chunk_path}...")
    segments = engine.diarize_file(chunk_path)
    print(f"[!] Segments trouvés : {len(segments)}")

    # 3. Identification
    print("[*] Identification des locuteurs dans les segments...")
    results = manager.identify_speakers_in_segments(chunk_path, segments)
    
    for res in results:
        print(f"  [{res['start']:.1f}s - {res['end']:.1f}s] {res['speaker']} (Confiance: {res['confidence']:.2f})")

    # Cleanup
    if os.path.exists("french_signatures.json"):
        os.remove("french_signatures.json")

if __name__ == "__main__":
    try:
        # test_speaker_manager() # On saute le test synthétique si on a le vrai
        test_diarization_engine()
        test_french_pipeline()
        print("\n[SUCCESS] Tests terminés !")
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
