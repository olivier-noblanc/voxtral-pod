import os
os.environ["DISABLE_NNPACK"] = "1"
os.environ["NNPACK_LOG_LEVEL"] = "0"
os.environ["OMP_NUM_THREADS"] = "1" # Help prevent initialization noise in some libs

import logging
from backend.config import setup_gpu, setup_warnings

# Immediate silence of NNPACK and other technical noise
setup_gpu()
setup_warnings()

# Configure global logging level (defaults to INFO, override with LOG_LEVEL=DEBUG if needed)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# --- Silence noisy external libraries ---
for noisy_logger in [
    "diarize.embeddings", 
    "wespeakerruntime", 
    "onnxruntime", 
    "urllib3", 
    "multipart", 
    "numba",
    "speechbrain",
    "speechbrain.utils.fetching",
    "speechbrain.utils.parameter_transfer",
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

# Import shared state
from backend.state import get_asr_engine
# Import API router
from backend.routes import api as api_module
# Expose TRANSCRIPTIONS_DIR for external monkey‑patching (tests modify backend.main.TRANSCRIPTIONS_DIR)
from backend.config import TRANSCRIPTIONS_DIR

# ----------------------------------------------------------------------
# Helper: format_transcription
# ----------------------------------------------------------------------
# The function is pure (no external dependencies) and converts a raw
# transcription line into an HTML snippet. It is used by the frontend
# to display formatted segments.
# Expected behaviour (as exercised by tests):
#   * Recognise lines of the form "[start -> end] [SPEAKER] text"
#   * Escape the text part to avoid HTML injection
#   * Emit spans with classes "segment-time", "segment-speaker",
#     and "segment-text"
#   * Fallback to a simple span with class "segment-text" when the
#     line does not match the expected pattern.
#   * Return an empty string for empty input.
# ----------------------------------------------------------------------
from backend.config import CLEANUP_RETENTION_DAYS
from backend.utils import format_transcription
from backend.cleanup import periodic_cleanup_task
import asyncio

@asynccontextmanager
async def lifespan(app):
    # startup
    from backend.state import init_db
    init_db()
    get_asr_engine(load_model=False)
    cleanup_task = asyncio.create_task(periodic_cleanup_task(CLEANUP_RETENTION_DAYS))
    yield
    # shutdown
    cleanup_task.cancel()

app = FastAPI(title="SOTA ASR Server", version="4.0.0", lifespan=lifespan)
# trusted_hosts : accepter uniquement les proxies locaux (nginx/caddy sur la même machine)
_TRUSTED_PROXIES = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=_TRUSTED_PROXIES)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include all routes defined in backend/routes/api.py
app.include_router(api_module.router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)