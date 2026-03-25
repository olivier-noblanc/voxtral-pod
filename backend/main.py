import os
import sys
import re
import html
import codecs
import shutil
import asyncio
import datetime

import torch
import boto3  # pyright: ignore[reportMissingImports]
from dulwich.repo import Repo  # pyright: ignore[reportMissingImports]
from dulwich import porcelain
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, RedirectResponse
from backend.config import setup_gpu, setup_warnings, TRANSCRIPTIONS_DIR, TEMP_DIR
from backend.html_ui import HTML_UI
# No route definitions in this file (all routes are defined in backend/routes/api.py)

def format_transcription(text: str) -> str:
    """
    Convertit le texte brut d’une transcription en HTML structuré.
    Le rendu final est plus lisible :
      • chaque segment est encapsulé dans ``<div class="segment">`` ;
      • les horodatages et les intervenants sont clairement séparés ;
      • le texte est correctement échappé pour éviter les injections HTML.
    Le format d’entrée accepte plusieurs notations :
      1. ``[hh:mm:ss] [Speaker] texte`` – horodatage + speaker ;
      2. ``[Speaker] texte`` – speaker uniquement (typique du live) ;
      3. ``Speaker: texte`` – format « Speaker: » ;
      4. texte brut – aucune métadonnée.
    """
    # Décoder les séquences \uXXXX littérales si présentes
    try:
        text = codecs.decode(text, "unicode_escape").encode("latin-1").decode("utf-8")
    except Exception:
        pass  # texte déjà décodé

    lines = text.split("\n")
    html_parts = []
    for line in lines:
        stripped = line.strip()
        # 1️⃣ [time] [speaker] texte
        match = re.match(r"^\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)$", stripped)
        if match:
            time_str, speaker, content = match.groups()
            html_parts.append(f'''
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-time">{html.escape(time_str)}</span>
                        <span class="segment-speaker" data-speaker="{html.escape(speaker)}">{html.escape(speaker)}</span>
                    </div>
                    <div class="segment-text">{html.escape(content)}</div>
                </div>
            ''')
            continue

        # 2️⃣ [speaker] texte (sans timestamp)
        match2 = re.match(r"^\[([^\]]+)\]\s*(.*)$", stripped)
        if match2:
            speaker, content = match2.groups()
            html_parts.append(f'''
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-speaker" data-speaker="{html.escape(speaker)}">{html.escape(speaker)}</span>
                    </div>
                    <div class="segment-text">{html.escape(content)}</div>
                </div>
            ''')
            continue

        # 3️⃣ speaker: texte
        match3 = re.match(r"^([^:]+):\s*(.*)$", stripped)
        if match3:
            speaker, content = match3.groups()
            html_parts.append(f'''
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-speaker" data-speaker="{html.escape(speaker.strip())}">{html.escape(speaker.strip())}</span>
                    </div>
                    <div class="segment-text">{html.escape(content.strip())}</div>
                </div>
            ''')
            continue

        # 4️⃣ texte brut
        if stripped:
            html_parts.append(f'''
                <div class="segment">
                    <div class="segment-text">{html.escape(stripped)}</div>
                </div>
            ''')
    # Enveloppe globale pour faciliter le style CSS
    return f'<div class="transcript">\n' + "\n".join(html_parts) + "\n</div>"
from backend.core.engine import SotaASR
from backend.core.live import LiveSession
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI(title="SOTA ASR Server", version="4.0.0")
app.add_middleware(ProxyHeadersMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Taille maximale du dictionnaire des jobs en mémoire (évite le memory leak)
JOBS_DB_MAX_SIZE = 500

# Global state
model_name = os.getenv("ASR_MODEL", "whisper").lower()
asr_engine = SotaASR(model_id=model_name, hf_token=os.environ.get("HF_TOKEN"))
jobs_db = {}

@app.on_event("startup")
async def startup_event():
    setup_gpu()
    setup_warnings()
    asr_engine.load()

# GET routes are defined in backend/routes/api.py

# Inclusion du router contenant toutes les routes POST
from backend.routes import api as api_module
api_router = api_module.router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)