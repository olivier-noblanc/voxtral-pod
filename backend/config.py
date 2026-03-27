import os
import warnings

_CPU_THREADS_CONFIGURED = False

# === CACHE HUGGINGFACE ===
_CACHE_DIR = os.path.abspath("hf_cache")
_HUB_DIR   = os.path.join(_CACHE_DIR, "hub")
os.makedirs(_HUB_DIR, exist_ok=True)

for _k, _v in [
    ("HF_HOME",              _CACHE_DIR),
    ("HF_HUB_CACHE",         _HUB_DIR),
    ("TRANSFORMERS_CACHE",   _HUB_DIR),
    ("HUGGINGFACE_HUB_CACHE",_HUB_DIR),
]:
    os.environ[_k] = _v

# === GPU PERFORMANCE OPTIMIZATION (Ampere+) ===
def setup_gpu():
    """
    Initialise les paramètres GPU/CPU de torch.
    Désactive NNPACK si demandé et configure les threads CPU si aucun GPU n'est disponible.
    """
    global _CPU_THREADS_CONFIGURED
    import torch  # lazy import

    # Désactivation de NNPACK si demandé. Certaines versions de torch n'exposent pas
    # l'attribut ``enabled`` ; on vérifie donc sa présence avant de l'utiliser.
    disable_nnpack = os.getenv("DISABLE_NNPACK", "0").lower() in ("1", "true", "yes", "on")
    if disable_nnpack and hasattr(torch.backends, "nnpack"):
        if hasattr(torch.backends.nnpack, "enabled"):
            # ``enabled`` peut ne pas être présent dans toutes les builds de torch.
            # Le ``# type: ignore`` indique à l'analyseur de type d'ignorer cette
            # possible absence d'attribut.
            torch.backends.nnpack.enabled = False  # type: ignore[attr-defined]
        elif hasattr(torch.backends.nnpack, "is_available"):
            # Aucun moyen direct de désactiver, on laisse le comportement par défaut.
            pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif not _CPU_THREADS_CONFIGURED:
        cpu_threads = max(1, int(os.getenv("CPU_THREADS", str(os.cpu_count() or 1))))
        torch.set_num_threads(cpu_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(4, cpu_threads // 2 or 1)))
        _CPU_THREADS_CONFIGURED = True

def get_vram_gb():
    """Returns the total VRAM of the primary GPU in GB. 0 if no GPU."""
    import torch  # lazy
    if not torch.cuda.is_available():
        return 0
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:
        return 0

# === WARNINGS FILTERS ===
def setup_warnings():
    # Ignore specific math warnings from pyannote on very short segments
    warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")
    # Ignore ReproducibilityWarning since we explicitly enable TF32
    warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorFloat-32.*")
    # Ignore torchcodec warnings
    warnings.filterwarnings("ignore", message=r".*torchcodec is not installed correctly.*", category=UserWarning)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTIONS_DIR = "transcriptions_terminees"
TEMP_DIR = "temp_batch"
CLEANUP_RETENTION_DAYS = int(os.getenv("CLEANUP_RETENTION_DAYS", "90"))

os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(TRANSCRIPTIONS_DIR, "live_audio"), exist_ok=True)
os.makedirs(os.path.join(TRANSCRIPTIONS_DIR, "batch_audio"), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
