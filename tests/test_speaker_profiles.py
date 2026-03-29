import os
import tempfile
import numpy as np
import pytest

# Ensure the test uses a temporary SQLite DB
def test_save_and_load_profiles():
    # Use a temporary directory for the DB file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_profiles.db")
        os.environ["SPEAKER_PROFILES_DB"] = db_path

        # Import after setting env var so the module picks up the path
        from backend.core import speaker_profiles

        # Clean start
        profiles = speaker_profiles.load_profiles()
        assert profiles == {}

        # Create a dummy embedding
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Save a profile
        speaker_profiles.save_profile("SPEAKER_00", "Olivier", emb)

        # Load and verify
        loaded = speaker_profiles.load_profiles()
        assert "SPEAKER_00" in loaded
        name, loaded_emb = loaded["SPEAKER_00"]
        assert name == "Olivier"
        assert loaded_emb is not None
        np.testing.assert_allclose(loaded_emb, emb)

def test_match_embedding_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_profiles.db")
        os.environ["SPEAKER_PROFILES_DB"] = db_path
        os.environ["VOICE_ID_THRESHOLD"] = "0.8"  # higher threshold for test

        from backend.core import speaker_profiles

        # Two profiles with distinct embeddings
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        speaker_profiles.save_profile("SPEAKER_A", "Alice", emb_a)
        speaker_profiles.save_profile("SPEAKER_B", "Bob", emb_b)

        # Query with an embedding close to Alice
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        match = speaker_profiles.match_embedding(query)
        # With threshold 0.8 the match should be Alice
        assert match is not None
        speaker_id, name = match
        assert speaker_id == "SPEAKER_A"
        assert name == "Alice"

        # Query with an embedding far from both (low similarity)
        far_query = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        match = speaker_profiles.match_embedding(far_query)
        # Should return None because similarity < threshold
        assert match is None