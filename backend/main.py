import os
from fastapi import FastAPI
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
from html import escape as html_escape
import re

def format_transcription(line: str) -> str:
    """
    Convert a raw transcription line into an HTML fragment.

    Parameters
    ----------
    line: str
        Raw line, e.g. "[0.50s -> 1.20s] [SPEAKER_00] Bonjour"

    Returns
    -------
    str
        HTML snippet containing the formatted segment.
    """
    if not line:
        return ""

    # Pattern with optional whitespace around components
    pattern = r"^\[([0-9.]+)s\s*->\s*([0-9.]+)s\]\s*\[([^\]]+)\]\s*(.*)$"
    match = re.match(pattern, line)
    if match:
        start, end, speaker, text = match.groups()
        # Escape the free‑form text to prevent HTML injection
        escaped_text = html_escape(text)
        return (
            f'<span class=\"segment-time\">[{start}s → {end}s]</span> '
            f'<span class=\"segment-speaker\">[{speaker}]</span> '
            f'<span class=\"segment-text\">{escaped_text}</span>'
        )
    else:
        # Fallback: plain text with generic class
        escaped = html_escape(line)
        return f'<span class=\"segment-text\">{escaped}</span>'

app = FastAPI(title="SOTA ASR Server", version="4.0.0")
app.add_middleware(ProxyHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize the ASR engine instance at startup."""
    # Instancie le modèle sans charger les poids pour économiser la VRAM si workers=N
    get_asr_engine(load_model=False)

# Include all routes defined in backend/routes/api.py
app.include_router(api_module.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)