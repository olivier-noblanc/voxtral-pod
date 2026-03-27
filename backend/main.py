import os
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