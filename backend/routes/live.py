from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import TRANSCRIPTIONS_DIR
from backend.routes.utils import _safe_join, _update_job_status
from backend.state import add_job, get_asr_engine, update_job

logger = logging.getLogger(__name__)
router = APIRouter()

async def _live_final_job(wav_path: str, job_id: str, client_id: str, timestamp: Optional[str]):
    """
    Process the final live audio file as a background job using Diarization.
    """
    print(f"[*] Starting Live Final Job for {job_id} (client: {client_id})")
    try:
        _update_job_status(job_id, "processing:Transcription avancée...", 5)
        engine = get_asr_engine(load_model=True)

        def progress_callback(step: str, pct: int):
            pct = max(0, min(100, pct))
            print(f"[*] [LiveFinal {job_id}] {step} : {pct}%")
            _update_job_status(job_id, f"processing:{step}", pct)

        result = await engine.process_file(wav_path, progress_callback=progress_callback)
        transcript = result.transcript

        client_dir = _safe_join(TRANSCRIPTIONS_DIR, client_id)
        os.makedirs(client_dir, exist_ok=True)
        ts = timestamp if timestamp is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_name = f"live_{ts}.txt"
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
            # Ne pas appeler clean_text automatiquement après chaque session live
            # Cela consomme du quota Albert inutilement et cause des 429 en cascade
            # from backend.core.postprocess import clean_text
            # cleaned = await clean_text(transcript)
            # if cleaned and cleaned.strip():
            #     cleaned_path = txt_path.replace(".txt", ".cleaned.md")
            #     with open(cleaned_path, "w", encoding="utf-8") as f:
            #         f.write(cleaned)
            # else:
            #     logger.warning(f"[Live {job_id}] Albert API returned empty cleanup.")
            pass
        except Exception as ce:
            logger.error(f"[Live {job_id}] Erreur nettoyage Albert auto: {ce}")

        add_job(job_id, {"status": "terminé", "progress": 100, "result_file": txt_name})
    except Exception as e:
        print(f"[!] Error in Live Final Job {job_id}: {e}")
        update_job(job_id, {"status": "erreur", "progress": 0, "error_details": str(e)})

@router.websocket("/live")
async def live_endpoint(
    websocket: WebSocket,
    client_id: str = "anonymous",
    partial_albert: bool = False,
    device_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_"):
        await websocket.close(code=4003)
        return
    # Register job early if session_id is provided
    if session_id:
        add_job(session_id, {"status": "en_attente", "progress": 0})
    await websocket.accept()
    from backend.core.live import LiveSession
    engine = get_asr_engine(load_model=True)
    session = LiveSession(engine, websocket, client_id, partial_albert=partial_albert, selected_audio_device_id=device_id)
    processor = asyncio.create_task(session.process_audio_queue())
    try:
        while True:
            data = await websocket.receive_bytes()
            await session.audio_queue.put(data)
    except WebSocketDisconnect:
        pass
    finally:
        await session.audio_queue.put(None)
        await processor
        # Toujours enregistrer un job si un session_id est fourni
        if session_id:
            job_id = session_id
            wav_path, timestamp = await session.save_wav_only()
            if wav_path:
                asyncio.create_task(_live_final_job(wav_path, job_id, client_id, timestamp))
            else:
                update_job(job_id, {"status": "terminé", "progress": 100, "result_file": None})
