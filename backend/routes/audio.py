from __future__ import annotations
import os
import shutil
import datetime
import logging
import json
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from backend.config import TEMP_DIR, TRANSCRIPTIONS_DIR
from backend.state import add_job, get_asr_engine, update_job
from backend.routes.utils import _safe_join, _validate_client_id, _update_job_status
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str) -> FileResponse:
    if f"_{client_id}_" not in filename:
        raise HTTPException(status_code=403, detail="Accès refusé : ce fichier n'appartient pas à votre session.")
    base_name = os.path.splitext(filename)[0]
    for subdir in ("live_audio", "batch_audio"):
        try:
            mp3_path = _safe_join(TRANSCRIPTIONS_DIR, subdir, base_name + ".mp3")
            if os.path.isfile(mp3_path):
                return FileResponse(
                    mp3_path,
                    media_type="audio/mpeg",
                    filename=base_name + ".mp3",
                    headers={"Content-Disposition": f"attachment; filename={base_name}.mp3"}
                )
            file_path = _safe_join(TRANSCRIPTIONS_DIR, subdir, filename)
            if os.path.isfile(file_path):
                return FileResponse(
                    file_path,
                    media_type="audio/wav",
                    filename=filename,
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        except HTTPException:
            continue
    raise HTTPException(status_code=404, detail="Fichier audio non trouvé.")

@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    client_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, str]:
    _validate_client_id(client_id)
    if chunk_index == 0:
        add_job(file_id, {"status": "uploading", "progress": 0})
    upload_dir = _safe_join(TEMP_DIR, file_id)
    os.makedirs(upload_dir, exist_ok=True)
    chunk_path = _safe_join(upload_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())
    if chunk_index == total_chunks - 1:
        _update_job_status(file_id, "processing:Réassemblage...", 0)
        assembled_path = os.path.join(upload_dir, "audio_full.tmp")
        await asyncio.to_thread(_assemble_chunks, assembled_path, total_chunks, upload_dir)
        background_tasks.add_task(_gpu_job, assembled_path, file_id, client_id)
    return {"status": "ok"}

def _assemble_chunks(assembled_path: str, total_chunks: int, upload_dir: str) -> None:
    with open(assembled_path, "wb") as out_f:
        for i in range(total_chunks):
            part = os.path.join(upload_dir, f"chunk_{i:04d}")
            with open(part, "rb") as in_f:
                out_f.write(in_f.read())
            os.remove(part)

async def _gpu_job(assembled_path: str, file_id: str, client_id: str) -> None:
    print(f"[*] Starting GPU Job for {file_id} (Client: {client_id})")
    try:
        _update_job_status(file_id, "processing:Transcription...", 10)
        engine = get_asr_engine(load_model=True)
        def progress_callback(step: str, pct: int):
            pct = max(0, min(100, pct))
            print(f"[*] [Batch {file_id}] {step} : {pct}%")
            _update_job_status(file_id, f"processing:{step}", pct)
        result = await engine.process_file(assembled_path, progress_callback=progress_callback)
        transcript = result.transcript
        client_dir = _safe_join(TRANSCRIPTIONS_DIR, client_id)
        os.makedirs(client_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_name = f"batch_{ts}.txt"
        txt_path = os.path.join(client_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        # Save RTTM and Diarization JSON for SSR viewer
        file_id_rttm = txt_name.replace(".txt", "")
        rttm_path = txt_path.replace(".txt", ".rttm")
        with open(rttm_path, "w", encoding="utf-8") as f:
            f.write(result.to_rttm(file_id_rttm))
        
        diar_json_path = txt_path.replace(".txt", ".diar.json")
        with open(diar_json_path, "w", encoding="utf-8") as f:
            json.dump(result.segments, f, indent=2)
        
        try:
            from backend.core.postprocess import clean_text
            cleaned = await clean_text(transcript)
            if cleaned and cleaned.strip():
                cleaned_path = txt_path.replace(".txt", ".cleaned.md")
                with open(cleaned_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            else:
                logger.warning(f"[Batch {file_id}] Albert API returned empty cleanup.")
        except Exception as ce:
            logger.error(f"[Batch {file_id}] Erreur nettoyage Albert auto: {ce}")

        add_job(file_id, {"status": "terminé", "progress": 100, "result_file": txt_name})
        try:
            batch_audio_dir = _safe_join(TRANSCRIPTIONS_DIR, "batch_audio")
            os.makedirs(batch_audio_dir, exist_ok=True)
            archived_audio_name = f"batch_{client_id}_{ts}.wav"
            archived_audio_path = _safe_join(batch_audio_dir, archived_audio_name)
            shutil.copy2(assembled_path, archived_audio_path)
        except Exception:
            pass
    except Exception as e:
        print(f"[!] Error in Transcription Job {file_id}: {e}")
        from backend.state import update_job
        update_job(file_id, {"status": "erreur", "progress": 0, "error_details": str(e)})
    finally:
        try:
            if os.path.exists(assembled_path):
                os.remove(assembled_path)
            upload_dir = os.path.dirname(assembled_path)
            if os.path.exists(upload_dir) and not os.listdir(upload_dir):
                os.rmdir(upload_dir)
        except Exception:
            pass
