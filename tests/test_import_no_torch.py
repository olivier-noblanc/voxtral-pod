"""
test_import_no_torch.py
-----------------------
Vérifie que les modules critiques s'importent sans erreur même quand torch
n'est pas accessible (simulation via sys.modules patching).

Régresse:
  - Bug #1 : indentation cassée dans TranscriptionEngine.__init__
  - Bug #2 : import torch top-level dans config.py
"""
import sys
import types
import importlib
import pytest


class _FakeTorch(types.ModuleType):
    """Stub minimal de torch pour les tests sans GPU."""
    class cuda:
        @staticmethod
        def is_available(): return False

    class backends:
        class nnpack:
            enabled = True
        class cuda:
            matmul = type("_", (), {"allow_tf32": False})()
        class cudnn:
            allow_tf32 = False

    @staticmethod
    def set_num_threads(n): pass
    @staticmethod
    def set_num_interop_threads(n): pass

    # Pour from_numpy & co utilisés ailleurs
    @staticmethod
    def from_numpy(x): return x


def _hide_torch():
    """Retire torch de sys.modules pour simuler son absence."""
    return sys.modules.pop("torch", None)


def _restore_torch(original):
    if original is not None:
        sys.modules["torch"] = original


# ── Bug #2 : config.py importable sans torch ────────────────────────────────

def test_config_importable_without_torch(monkeypatch):
    """config.py ne doit pas importer torch au niveau module."""
    original = _hide_torch()
    try:
        # Forcer rechargement propre
        for mod in list(sys.modules):
            if mod == "backend.config" or mod.startswith("backend.config."):
                del sys.modules[mod]

        # Ne doit pas lever ImportError
        import backend.config  # noqa: F401
    finally:
        _restore_torch(original)


# ── Bug #1 : TranscriptionEngine instanciable sans torch ────────────────────

def test_transcription_engine_init_without_torch(monkeypatch):
    """TranscriptionEngine doit s'instancier (device='cpu') même sans torch."""
    # On injecte un stub qui signale l'absence de CUDA
    fake = _FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake)

    # Nettoyer le cache du module pour forcer le re-import
    for mod in list(sys.modules):
        if "backend.core.transcription" in mod:
            del sys.modules[mod]

    from backend.core.transcription import TranscriptionEngine  # noqa: PLC0415
    engine = TranscriptionEngine(model_id="whisper")

    assert engine.device == "cpu", f"expected 'cpu', got {engine.device!r}"
    assert engine.model_id == "whisper"


def test_transcription_engine_device_param():
    """Un device explicite doit être respecté sans appel à torch."""
    for mod in list(sys.modules):
        if "backend.core.transcription" in mod:
            del sys.modules[mod]

    from backend.core.transcription import TranscriptionEngine  # noqa: PLC0415
    engine = TranscriptionEngine(model_id="whisper", device="cpu")
    assert engine.device == "cpu"
