from __future__ import annotations
import os
import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from backend.config import TRANSCRIPTIONS_DIR
from backend.routes.utils import _safe_join, _validate_client_id, markdown, templates

logger = logging.getLogger(__name__)
router = APIRouter()

def _render_postprocess_page(request: Request, title: str, content: str, filename: str) -> Response:
    rendered_content = markdown.markdown(content, extensions=['extra', 'nl2br'])
    template = templates.get_template("postprocess.html")
    html = template.render({
        "title": title,
        "filename": filename,
        "raw_content": content,
        "rendered_content": rendered_content,
        "request": request,
    })
    return HTMLResponse(html)

@router.post("/summary/{filename}")
async def generate_summary(filename: str, client_id: str) -> Dict[str, Any]:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import summarize_text
    try:
        summary = await summarize_text(content)
        return {"status": "ok", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/actions/{filename}")
async def generate_actions(filename: str, client_id: str) -> Dict[str, Any]:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import extract_actions_text
    try:
        actions_list = await extract_actions_text(content)
        actions_text = "\n".join(f"- {a}" for a in actions_list)
        return {"status": "ok", "actions": actions_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup/{filename}")
async def generate_cleanup(filename: str, client_id: str) -> Dict[str, Any]:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import clean_text
    try:
        cleaned = await clean_text(content)
        return {"status": "ok", "cleanup": cleaned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_summary/{client_id}/{filename}")
async def view_summary(client_id: str, filename: str, req: Request) -> Response:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import summarize_text
    try:
        summary = await summarize_text(content)
        return _render_postprocess_page(req, "Compte Rendu Albert", summary, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_actions/{client_id}/{filename}")
async def view_actions(client_id: str, filename: str, req: Request) -> Response:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import extract_actions_text
    try:
        actions_list = await extract_actions_text(content)
        actions_text = "\n".join(f"- {a}" for a in actions_list)
        return _render_postprocess_page(req, "Actions Albert", actions_text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_cleanup/{client_id}/{filename}")
async def view_cleanup(client_id: str, filename: str, req: Request) -> Response:
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import clean_text
    try:
        cleaned = await clean_text(content)
        return _render_postprocess_page(req, "Nettoyage Albert", cleaned, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
