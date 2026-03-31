from __future__ import annotations
import os
import pathlib
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from backend.config import TRANSCRIPTIONS_DIR
from backend.utils import format_transcription
from backend.routes.utils import _safe_join, markdown, templates

router = APIRouter()

@router.get("/transcriptions")
async def list_trans(client_id: str) -> List[Dict[str, Any]]:
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id)
    if not os.path.isdir(p):
        return []
    files = sorted([f for f in os.listdir(p) if f.endswith(".txt")], reverse=True)
    result = []
    for f in files:
        has_cleanup = os.path.isfile(os.path.join(p, f.replace(".txt", ".cleaned.md")))
        result.append({"name": f, "has_cleanup": has_cleanup})
    return result

@router.get("/transcriptions/")
async def list_trans_slash(client_id: str) -> List[Dict[str, Any]]:
    return await list_trans(client_id)

@router.delete("/transcriptions/{filename}")
async def delete_trans(filename: str, client_id: str) -> Dict[str, str]:
    trans_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(trans_path):
        raise HTTPException(status_code=404, detail="Fichier transcription introuvable.")
    if os.path.isfile(trans_path):
        os.remove(trans_path)

    # Supprimer les versions nettoyées, RTTM et JSON de diarisation
    base_name = pathlib.Path(filename).stem
    for ext in (".cleaned.md", ".rttm", ".diar.json"):
        extra_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, base_name + ext)
        if os.path.isfile(extra_path):
            os.remove(extra_path)

    # Supprimer l'audio associé (WAV et MP3)
    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_name_base = f"batch_{client_id}_{ts}"
        audio_dir = "batch_audio"
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_name_base = f"live_{client_id}_{ts}"
        audio_dir = "live_audio"
    
    for ext in (".wav", ".mp3"):
        audio_path = _safe_join(TRANSCRIPTIONS_DIR, audio_dir, audio_name_base + ext)
        if os.path.isfile(audio_path):
            os.remove(audio_path)

    return {"status": "ok"}

@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str) -> FileResponse:
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type="text/plain", filename=filename)

@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str) -> Response:
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fichier transcript introuvable.")
    with open(file_path, "rb") as f:
        content = f.read()
    return Response(
        content,
        media_type="text/plain",
        headers={
            "Content-Type": "text/plain",
            "Content-Disposition": f"attachment; filename={filename}",
        },
    )

@router.get("/view/{client_id}/{filename}")
async def view_transcription(client_id: str, filename: str, req: Request) -> Response:
    client_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    cleaned_path = client_path.replace(".txt", ".cleaned.md")
    
    # Determine which file to serve
    if os.path.isfile(cleaned_path):
        file_path = cleaned_path
        is_cleaned = True
        is_temp = False
    elif os.path.isfile(client_path):
        file_path = client_path
        is_cleaned = False
        is_temp = False
    else:
        temp_path = _safe_join(TRANSCRIPTIONS_DIR, "live_audio", filename)
        if os.path.isfile(temp_path):
            if "_user_" in filename and f"_{client_id}_" not in filename:
                raise HTTPException(status_code=403, detail="Accès refusé.")
            file_path = temp_path
            is_cleaned = False
            is_temp = True
        else:
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if is_cleaned:
        # Render cleaned markdown
        rendered_content = markdown.markdown(content, extensions=['extra', 'nl2br'])
        template = templates.get_template("postprocess.html")
        html = template.render({
            "title": "Version Nettoyée Albert",
            "filename": filename,
            "raw_content": content,
            "rendered_content": rendered_content,
            "request": req,
        })
        return HTMLResponse(html)
    
    # Regular transcription view
    segments_html = "\n".join(format_transcription(line) for line in content.splitlines() if line.strip())
    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_filename = f"batch_{client_id}_{ts}.wav"
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_filename = f"live_{client_id}_{ts}.wav"
    audio_url = f"/download_audio/{client_id}/{audio_filename}"
    
    template = templates.get_template("view.html")
    html = template.render({
        "filename": filename,
        "audio_url": audio_url,
        "is_temp": is_temp,
        "segments_html": segments_html,
        "request": req,
    })
    return HTMLResponse(html)
