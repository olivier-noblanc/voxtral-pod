import os
import torch
import warnings

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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def get_vram_gb():
    """Returns the total VRAM of the primary GPU in GB. 0 if no GPU."""
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

os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
