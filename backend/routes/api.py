import asyncio
import os
import sys
import datetime

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
import boto3
from dulwich.repo import Repo
from dulwich import porcelain

# Import shared objects from main
from backend.main import app, asr_engine, model_name, jobs_db, TRANSCRIPTIONS_DIR, TEMP_DIR, SotaASR, JOBS_DB_MAX_SIZE

router = APIRouter()

@router.post("/git_update")
async def git_update():
    try:
        repo = Repo(".")

        porcelain.fetch(repo, "origin")

        all_refs = repo.get_refs()
        remote_branch_ref = b"refs/remotes/origin/main"
        if remote_branch_ref not in all_refs:
            remote_branch_ref = b"refs/remotes/origin/master"

        remote_sha = all_refs[remote_branch_ref]

        local_ref = b"refs/heads/main"
        if local_ref not in repo.refs:
            local_ref = b"refs/heads/master"
        repo.refs[local_ref] = remote_sha
        porcelain.reset(repo, b"HEAD", b"hard")

        async def _restart():
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(_restart())
        return {"stdout": "Mise à jour OK, redémarrage...", "stderr": ""}

    except Exception as e:
        return {"stdout": f"Erreur: {str(e)}", "stderr": ""}

@router.post("/change_model")
async def change_model_route(model: str):
    global asr_engine, model_name
    model_name = model
    asr_engine = SotaASR(model_id=model, hf_token=os.environ.get("HF_TOKEN"))
    asr_engine.load()
    return {"status": "ok"}

@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...), client_id: str = Form("anonymous"),
    chunk_index: int = Form(...), total_chunks: int = Form(...),
    file: UploadFile = File(...)
):
    if chunk_index == 0:
        jobs_db[file_id] = {"status": "uploading", "progress": 0}
        if len(jobs_db) > JOBS_DB_MAX_SIZE:
            oldest_keys = list(jobs_db.keys())[: len(jobs_db) - JOBS_DB_MAX_SIZE]
            for k in oldest_keys:
                jobs_db.pop(k, None)

    upload_dir = os.path.join(TEMP_DIR, file_id)
    os.makedirs(upload_dir, exist_ok=True)

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}")
    with open(chunk_path, "wb") as f:
        f.write(await file.read())

    if chunk_index == total_chunks - 1:
        jobs_db[file_id] = {"status": "processing:Réassemblage...", "progress": 0}
        assembled_path = os.path.join(upload_dir, "audio_full")
        with open(assembled_path, "wb") as out_f:
            for i in range(total_chunks):
                cp = os.path.join(upload_dir, f"chunk_{i:04d}")
                with open(cp, "rb") as in_f: out_f.write(in_f.read())
                if os.path.exists(cp): os.remove(cp)

        background_tasks.add_task(run_batch_job, assembled_path, file_id, client_id)

    return {"status": "ok"}

@router.post("/cancel/{file_id}")
async def cancel_route(file_id: str):
    if file_id in jobs_db:
        jobs_db[file_id] = {"status": "cancelled"}
    return {"status": "ok"}

@router.post("/save_live_transcription/{client_id}")
async def save_live_transcription_route(client_id: str, content: str = Form(...)):
    out_dir = os.path.join(TRANSCRIPTIONS_DIR, client_id)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"live_{ts}.txt"
    file_path = os.path.join(out_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "filename": filename}

@router.post("/update_transcription/{client_id}/{filename}")
async def update_transcription_route(client_id: str, filename: str, content: str = Form(...)):
    file_path = os.path.join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return RedirectResponse(url=f"/view/{client_id}/{filename}", status_code=303)

@router.post("/upload_s3")
async def upload_s3_route(
    filename: str = Form(...), content: str = Form(...),
    endpoint: str = Form(...), bucket: str = Form(...),
    access_key: str = Form(...), secret_key: str = Form(...)
):
    if boto3 is None:
        raise HTTPException(status_code=500, detail="Le module boto3 n'est pas installé.")
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        s3.put_object(Bucket=bucket, Key=filename, Body=content.encode("utf-8"), ContentType="text/plain; charset=utf-8")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))