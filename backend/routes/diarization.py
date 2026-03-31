from __future__ import annotations
import os
import pathlib
import json
from typing import Any
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from backend.config import TRANSCRIPTIONS_DIR
from backend.routes.utils import _safe_join, _validate_client_id, templates

router = APIRouter()

@router.get("/download_rttm/{client_id}/{filename}")
async def download_rttm(client_id: str, filename: str) -> FileResponse:
    """
    Télécharge le fichier RTTM correspondant à une transcription.
    """
    _validate_client_id(client_id)
    # The filename passed is the base name or .txt name
    base_name = pathlib.Path(filename).stem
    rttm_filename = f"{base_name}.rttm"
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, rttm_filename)
    
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fichier RTTM introuvable.")
    return FileResponse(path=file_path, filename=rttm_filename, media_type="text/plain")

@router.get("/diarization_data/{client_id}/{filename}")
async def get_diarization_data(client_id: str, filename: str) -> Any:
    """
    Retourne les données de diarisation brutes au format JSON.
    """
    _validate_client_id(client_id)
    base_name = pathlib.Path(filename).stem
    json_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, f"{base_name}.diar.json")
    
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Données de diarisation introuvables.")
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/view_diarization/{client_id}/{filename}")
async def view_diarization(client_id: str, filename: str, req: Request) -> HTMLResponse:
    """
    Affiche un tableau HTML (SSR via Jinja2) des segments de diarisation.
    """
    _validate_client_id(client_id)
    base_name = pathlib.Path(filename).stem
    json_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, f"{base_name}.diar.json")
    
    if not os.path.isfile(json_path):
        # Fallback: essayer de reconstruire à partir du RTTM si le JSON n'existe pas encore
        rttm_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, f"{base_name}.rttm")
        if not os.path.isfile(rttm_path):
            raise HTTPException(status_code=404, detail="Données de diarisation introuvables.")
        
        # Simple RTTM parser fallback
        segments = []
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segments.append({
                        "start": start,
                        "end": start + duration,
                        "speaker": speaker,
                        "text": "" # No snippet in RTTM
                    })
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

    template = templates.get_template("diarization_view.html")
    html = template.render({
        "segments": segments,
        "filename": base_name,
        "client_id": client_id,
        "request": req,
    })
    return HTMLResponse(html)

@router.post("/diarize")
async def api_diarize(file: UploadFile = File(...)) -> Any:
    """
    Endpoint de diarisation batch simple.
    """
    from backend.core.diarization_cpu import LightDiarizationEngine
    import tempfile
    
    suffix = pathlib.Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        engine = LightDiarizationEngine()
        engine.load()
        segments = engine.diarize_file(tmp_path)
        return segments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/benchmark")
async def api_benchmark(file: UploadFile = File(...)) -> Any:
    """
    Endpoint de benchmark performance pour la diarisation CPU.
    """
    from backend.core.diarization_cpu import LightDiarizationEngine
    import tempfile
    
    suffix = pathlib.Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        engine = LightDiarizationEngine()
        engine.load()
        stats = engine.benchmark(tmp_path)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
