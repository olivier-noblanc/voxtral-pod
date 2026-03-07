import numpy as np
from vad_manager import VADManager, SAMPLE_RATE

def test_vad_basic():
    print("[*] Test VADManager : Initialisation...")
    vad = VADManager(silero_sensitivity=0.4)
    
    # 1. Test Silence (zeros)
    print("[*] Test 1: Silence complet...")
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()
    is_speech = vad.is_speech(silence)
    print(f"    -> Speech détecté : {is_speech} (Attendu: False)")
    assert not is_speech
    
    # 2. Test Bruit Rose/Blanc (ne devrait pas être de la parole)
    print("[*] Test 2: Bruit blanc...")
    noise = (np.random.rand(SAMPLE_RATE) * 1000).astype(np.int16).tobytes()
    is_speech = vad.is_speech(noise)
    print(f"    -> Speech détecté : {is_speech} (Attendu: False ou très court)")

    # Note: On ne peut pas facilement générer de la parole synthétique sans librairie lourde,
    # mais ce test valide au moins que le moteur Silero/WebRTC est chargé.
    print("[✅] Test VAD de base réussi.")

if __name__ == "__main__":
    try:
        test_vad_basic()
    except Exception as e:
        print(f"[❌] Erreur pendant le test : {e}")
