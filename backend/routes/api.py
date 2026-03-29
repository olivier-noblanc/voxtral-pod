import os
import pathlib
import importlib.util
import numpy as np
import datetime
import asyncio
import shutil
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates
import re
import json
import logging
from backend.core.engine import TranscriptionResult

logger = logging.getLogger(__name__)

# Markdown handling: use real markdown if available, otherwise fallback
_markdown_spec = importlib.util.find_spec("markdown")
if _markdown_spec is not None:
    import markdown as _markdown_lib
    markdown = _markdown_lib
else:
    class _FallbackMarkdown:
        @staticmethod
        def markdown(text, **kwargs):
            """
            Very small markdown subset renderer handling headings, lists, and bold.
            """
            # Convert **bold** to <strong> tags
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

            lines = text.splitlines()
            html_lines = []
            in_list = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("### "):
                    if in_list:
                        html_lines.append("</ul>")
                        in_list = False
                    html_lines.append(f"<h3>{stripped[4:].strip()}</h3>")
                elif stripped.startswith("- "):
                    if not in_list:
                        html_lines.append("<ul>")
                        in_list = True
                    html_lines.append(f"<li>{stripped[2:].strip()}</li>")
                else:
                    if in_list:
                        html_lines.append("</ul>")
                        in_list = False
                    html_lines.append(line)
            if in_list:
                html_lines.append("</ul>")
            return "\n".join(html_lines)
    markdown = _FallbackMarkdown()

# Import shared state
from backend.state import get_job, update_job, add_job, JOBS_DB_MAX_SIZE, get_asr_engine, get_current_model, set_current_model
from backend.config import TEMP_DIR, TRANSCRIPTIONS_DIR, BASE_DIR
from backend.utils import format_transcription

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "backend", "templates"))

# ---------- Sécurité : anti path traversal ----------
def _safe_join(base: str, *parts: str) -> str:
    """
    Résout le chemin final et vérifie qu'il reste sous `base`.
    Lève HTTPException 400 en cas de tentative de path traversal.
    """
    base_resolved = pathlib.Path(base).resolve()
    target = (base_resolved / pathlib.Path(*parts)).resolve()
    if not str(target).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Chemin invalide.")
    return str(target)

def _validate_client_id(client_id: str):
    """Vérifie que le client_id est valide (format user_xxxxxxxx)."""
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_") or len(client_id) < 6:
        raise HTTPException(status_code=400, detail="Identifiant client (UUID) invalide ou anonyme non autorisé.")

def _require_admin_key(request: Request):
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        return
    provided = request.headers.get("X-Admin-Key", "")
    if provided != admin_key:
        raise HTTPException(status_code=403, detail="Clé admin invalide ou manquante.")

# ---------- Helper ----------
def _update_job_status(job_id: str, status: str, progress: int = 0, result_file: str | None = None):
    data = {"status": status, "progress": progress}
    if result_file:
        data["result_file"] = result_file
    update_job(job_id, data)

# ---------- GET routes ----------
@router.get("/")
async def home(request: Request):
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:7]
    except Exception:
        commit = "unknown"
    
    current_model = get_current_model()
    if os.getenv("TESTING") == "1":
        current_model = "mock"
    
    engine = get_asr_engine(load_model=False)
    
    if engine.no_gpu:
        model_options = '<option value="albert" selected>API Albert (Étalab)</option>'
    else:
        model_options = (
            '<option value="whisper">Faster-Whisper Large-v3</option>'
            '<option value="voxtral">Voxtral Mini 4B</option>'
            '<option value="albert">API Albert (Étalab)</option>'
            '<option value="mock">MODE TEST (Mock ASR)</option>'
        )
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "model_name": current_model,
            "device": "cuda" if not engine.no_gpu else "cpu",
            "commit": commit,
            "model_options": model_options
        }
    )

@router.get("/transcriptions")
async def list_trans(client_id: str):
    base = pathlib.Path(TRANSCRIPTIONS_DIR).resolve()
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
async def list_trans_slash(client_id: str):
    return await list_trans(client_id)

@router.delete("/transcriptions/{filename}")
async def delete_trans(filename: str, client_id: str):
    trans_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(trans_path):
        raise HTTPException(status_code=404, detail="Fichier transcription introuvable.")
    os.remove(trans_path)

    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_name = f"batch_{client_id}_{ts}.wav"
        audio_path = _safe_join(TRANSCRIPTIONS_DIR, "batch_audio", audio_name)
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_name = f"live_{client_id}_{ts}.wav"
        audio_path = _safe_join(TRANSCRIPTIONS_DIR, "live_audio", audio_name)
    
    if os.path.isfile(audio_path):
        os.remove(audio_path)

    return {"status": "ok"}

@router.get("/transcription/{filename}")
async def get_trans(filename: str, client_id: str):
    p = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(p):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type="text/plain", filename=filename)

@router.post("/summary/{filename}")
async def generate_summary(filename: str, client_id: str):
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
async def generate_actions(filename: str, client_id: str):
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
async def generate_cleanup(filename: str, client_id: str):
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

@router.get("/download_audio/{client_id}/{filename}")
async def download_audio(client_id: str, filename: str):
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

@router.get("/download_transcript/{client_id}/{filename}")
async def download_transcript(client_id: str, filename: str):
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
async def view_transcription(client_id: str, filename: str, request: Request):
    client_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    cleaned_path = client_path.replace(".txt", ".cleaned.md")
    
    show_cleaned = False
    if os.path.isfile(cleaned_path):
        file_path = cleaned_path
        show_cleaned = True
        is_temp = False
    elif os.path.isfile(client_path):
        file_path = client_path
        is_temp = False
    else:
        temp_path = _safe_join(TRANSCRIPTIONS_DIR, "live_audio", filename)
        if os.path.isfile(temp_path):
            if f"_user_" in filename and f"_{client_id}_" not in filename:
                raise HTTPException(status_code=403, detail="Accès refusé.")
            file_path = temp_path
            is_temp = True
        else:
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
            
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if show_cleaned:
        # Render markdown for the cleaned version
        rendered_content = markdown.markdown(content, extensions=['extra', 'nl2br'])
        return templates.TemplateResponse(
            request,
            "postprocess.html",
            {
                "title": "Version Nettoyée Albert",
                "filename": filename,
                "raw_content": content,
                "rendered_content": rendered_content
            }
        )

    segments_html = "\n".join(format_transcription(line) for line in content.splitlines() if line.strip())
    
    if filename.startswith("batch_"):
        ts = filename.replace("batch_", "").replace(".txt", "")
        audio_filename = f"batch_{client_id}_{ts}.wav"
    else:
        ts = filename.replace("live_", "").replace(".txt", "")
        audio_filename = f"live_{client_id}_{ts}.wav"
    
    audio_url = f"/download_audio/{client_id}/{audio_filename}"
    
    return templates.TemplateResponse(
        request,
        "view.html",
        {
            "filename": filename,
            "audio_url": audio_url,
            "is_temp": is_temp,
            "segments_html": segments_html
        }
    )

def _render_postprocess_page(request: Request, title: str, content: str, filename: str) -> Response:
    rendered_content = markdown.markdown(content, extensions=['extra', 'nl2br'])
    return templates.TemplateResponse(
        request,
        "postprocess.html",
        {
            "title": title,
            "filename": filename,
            "raw_content": content,
            "rendered_content": rendered_content
        }
    )

@router.get("/view_summary/{client_id}/{filename}")
async def view_summary(client_id: str, filename: str, request: Request):
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import summarize_text
    try:
        summary = await summarize_text(content)
        return _render_postprocess_page(request, "Compte Rendu Albert", summary, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_rttm/{client_id}/{filename}")
async def download_rttm(client_id: str, filename: str):
    """
    Télécharge le fichier RTTM correspondant à une transcription.
    """
    _validate_client_id(client_id)
    # The filename passed is the base name or .txt name
    base_name = filename.replace(".txt", "").replace(".cleaned.md", "")
    rttm_filename = f"{base_name}.rttm"
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, rttm_filename)
    
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Fichier RTTM introuvable.")
    return FileResponse(path=file_path, filename=rttm_filename, media_type="text/plain")

@router.get("/diarization_data/{client_id}/{filename}")
async def get_diarization_data(client_id: str, filename: str):
    """
    Retourne les données de diarisation brutes au format JSON.
    """
    _validate_client_id(client_id)
    base_name = filename.replace(".txt", "").replace(".cleaned.md", "")
    json_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, f"{base_name}.diar.json")
    
    if not os.path.isfile(json_path):
        raise HTTPException(status_code=404, detail="Données de diarisation introuvables.")
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/view_diarization/{client_id}/{filename}")
async def view_diarization(client_id: str, filename: str, request: Request):
    """
    Affiche un tableau HTML (SSR via Jinja2) des segments de diarisation.
    """
    _validate_client_id(client_id)
    base_name = filename.replace(".txt", "").replace(".cleaned.md", "")
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


    return templates.TemplateResponse("diarization_view.html", {
        "request": request,
        "segments": segments,
        "filename": base_name,
        "client_id": client_id
    })


@router.get("/view_actions/{client_id}/{filename}")
async def view_actions(client_id: str, filename: str, request: Request):
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
        return _render_postprocess_page(request, "Actions Albert", actions_text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view_cleanup/{client_id}/{filename}")
async def view_cleanup(client_id: str, filename: str, request: Request):
    _validate_client_id(client_id)
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    from backend.core.postprocess import clean_text
    try:
        cleaned = await clean_text(content)
        return _render_postprocess_page(request, "Nettoyage Albert", cleaned, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{file_id}")
async def status_route(file_id: str):
    return get_job(file_id, {"status": "not_found"})

@router.get("/git_status")
async def git_status():
    try:
        import subprocess
        subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, check=False)
        local_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        behind_out = subprocess.check_output(["git", "rev-list", "--count", "HEAD..origin/main"], text=True).strip()
        behind = int(behind_out) if behind_out.isdigit() else 0
        return {"commit": local_sha[:7], "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}

# ---------- Speaker profile endpoints ----------
from typing import List, Union
@router.get("/speaker_profiles")
async def get_speaker_profiles() -> List[dict]:
    import backend.core.speaker_profiles as speaker_profiles
    profiles = speaker_profiles.load_profiles()
    return [{"speaker_id": sid, "name": name} for sid, (name, _) in profiles.items()]

@router.post("/speaker_profiles")
async def upsert_speaker_profile(payload: dict) -> Union[dict, None]:
    import backend.core.speaker_profiles as speaker_profiles
    speaker_id = payload.get("speaker_id")
    name = payload.get("name")
    embedding = payload.get("embedding")
    if not speaker_id or not name:
        raise HTTPException(status_code=400, detail="speaker_id and name are required.")
    emb_array = None
    if embedding is not None:
        try:
            import numpy as np
            emb_array = np.array(embedding, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding format.")
    speaker_profiles.save_profile(speaker_id, name, emb_array)
    return {"status": "ok"}

def _extract_and_save_profile_sync(audio_path: str, start_s: float, end_s: float, speaker_id: str):
    import os
    if not os.path.isfile(audio_path):
        print(f"[!] Audio file {audio_path} not found for profile extraction.")
        return
    try:
        from resemblyzer import preprocess_wav, VoiceEncoder
        wav = preprocess_wav(audio_path)
        sample_rate = 16000
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        segment_audio = wav[start_idx:end_idx]
        if len(segment_audio) < 16000 * 0.5:
            print("[!] Audio segment too short to extract reliable voice profile.")
            return
        encoder = VoiceEncoder("cpu")
        emb = encoder.embed_utterance(segment_audio)
        import backend.core.speaker_profiles as speaker_profiles
        from typing import cast
        speaker_profiles.save_profile(speaker_id=speaker_id, name=speaker_id, embedding=cast(np.ndarray, emb))
        print(f"[*] Saved voice profile for speaker: {speaker_id}")
    except Exception as e:
        print(f"[!] Warning: Failed to extract speaker embedding for {speaker_id}: {e}")

@router.post("/segment_update")
async def update_segment_speaker(payload: dict, background_tasks: BackgroundTasks):
    import re
    client_id = payload.get("client_id")
    filename = payload.get("filename")
    segment_index = payload.get("segment_index")
    new_speaker = payload.get("new_speaker")
    # Explicitly check for None to satisfy Pyright
    if client_id is None or filename is None or new_speaker is None or not isinstance(segment_index, int):
        raise HTTPException(status_code=400, detail="Invalid payload: missing or invalid fields.")
    
    file_path = _safe_join(TRANSCRIPTIONS_DIR, client_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Transcription not found.")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if segment_index < 0 or segment_index >= len(lines):
        raise HTTPException(status_code=400, detail="segment_index out of range.")
    match = re.search(r'\[([\d.]+)s\s*->\s*([\d.]+)s\]', lines[segment_index])
    if match:
        start_s = float(match.group(1))
        end_s = float(match.group(2))
        if filename.startswith("batch_"):
            ts = filename.replace("batch_", "").replace(".txt", "")
            audio_name = f"batch_{client_id}_{ts}.wav"
            audio_dir = "batch_audio"
        else:
            ts = filename.replace("live_", "").replace(".txt", "")
            audio_name = f"live_{client_id}_{ts}.wav"
            audio_dir = "live_audio"
        audio_path = _safe_join(TRANSCRIPTIONS_DIR, audio_dir, audio_name)
        background_tasks.add_task(_extract_and_save_profile_sync, audio_path, start_s, end_s, new_speaker)
    new_line = re.sub(
        r'(\[[^\]]*s\s*->\s*[^\]]*s\]\s*)\[[^\]]*\]',
        rf'\1[{new_speaker}]',
        lines[segment_index]
    )
    if new_line == lines[segment_index]:
        raise HTTPException(status_code=400, detail="Format de ligne non reconnu (timestamp+speaker attendus).")
    lines[segment_index] = new_line
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return {"status": "ok"}
    
# ---------- New CPU Diarization & Speaker ID Endpoints ----------

@router.post("/diarize")
async def api_diarize(file: UploadFile = File(...)):
    """
    Endpoint de diarisation batch simple.
    """
    from backend.core.diarization_cpu import LightDiarizationEngine
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(file.filename or "audio.wav").suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        engine = LightDiarizationEngine()
        # No load() needed for functional API, but kept for interface compatibility
        engine.load()
        segments = engine.diarize_file(tmp_path)
        return segments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/benchmark")
async def api_benchmark(file: UploadFile = File(...)):
    """
    Endpoint de benchmark performance pour la diarisation CPU.
    """
    from backend.core.diarization_cpu import LightDiarizationEngine
    import tempfile
    
    suffix = pathlib.Path(file.filename).suffix if file.filename else ".wav"
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

@router.post("/speakers/enroll")
async def api_enroll_speaker(name: str = Form(...), file: UploadFile = File(...)):
    """
    Enrôle un nouveau locuteur (signature vocale).
    """
    from backend.core.speaker_manager import SpeakerManager
    import tempfile
    
    suffix = pathlib.Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        manager = SpeakerManager()
        success = manager.enroll_speaker(name, tmp_path)
        if success:
            return {"status": "ok", "message": f"Locuteur {name} enrôlé avec succès."}
        else:
            raise HTTPException(status_code=500, detail="Échec de l'enrôlement.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/speakers/identify")
async def api_identify_speakers(file: UploadFile = File(...)):
    """
    Identifie les locuteurs connus dans un fichier audio segmenté par diarisation.
    """
    from backend.core.diarization_cpu import LightDiarizationEngine
    from backend.core.speaker_manager import SpeakerManager
    import tempfile
    
    suffix = pathlib.Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        # 1. Diarisation pour obtenir les segments
        diar_engine = LightDiarizationEngine()
        diar_engine.load()
        segments = diar_engine.diarize_file(tmp_path)
        
        # 2. Identification des locuteurs sur chaque segment
        manager = SpeakerManager()
        results = manager.identify_speakers_in_segments(tmp_path, segments)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------- POST routes ----------
@router.post("/git_update")
async def git_update(request: Request):
    _require_admin_key(request)
    try:
        import subprocess
        subprocess.run(["git", "fetch", "origin", "main"], check=True)
        res = subprocess.run(["git", "reset", "--hard", "origin/main"], capture_output=True, text=True, check=True)
        async def _restart():
            await asyncio.sleep(2)
            print("[INFO] Git CLI update OK. Exiting container...")
            os._exit(0)
        asyncio.create_task(_restart())
        return {"stdout": res.stdout or "Mise à jour réussie. Redémarrage...", "stderr": ""}
    except Exception as e:
        return {"stdout": "", "stderr": str(e)}

@router.post("/change_model")
async def change_model_route(model: str, request: Request):
    _require_admin_key(request)
    set_current_model(model.lower())
    get_asr_engine(load_model=True)
    return {"status": "ok"}

@router.post("/cleanup")
async def trigger_cleanup(request: Request):
    _require_admin_key(request)
    from backend.cleanup import run_cleanup
    from backend.config import CLEANUP_RETENTION_DAYS
    try:
        stats = await asyncio.to_thread(run_cleanup, CLEANUP_RETENTION_DAYS)
        return {"status": "ok", "deleted": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_chunk")
async def batch_chunk_route(
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    client_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file: UploadFile = File(...),
):
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

async def _gpu_job(assembled_path: str, file_id: str, client_id: str):
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

async def _live_final_job(wav_path: str, job_id: str, client_id: str, timestamp: str | None):
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
            from backend.core.postprocess import clean_text
            cleaned = await clean_text(transcript)
            if cleaned and cleaned.strip():
                cleaned_path = txt_path.replace(".txt", ".cleaned.md")
                with open(cleaned_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            else:
                logger.warning(f"[Live {job_id}] Albert API returned empty cleanup.")
        except Exception as ce:
            logger.error(f"[Live {job_id}] Erreur nettoyage Albert auto: {ce}")

        add_job(job_id, {"status": "terminé", "progress": 100, "result_file": txt_name})
    except Exception as e:
        print(f"[!] Error in Live Final Job {job_id}: {e}")
        from backend.state import update_job
        update_job(job_id, {"status": "erreur", "progress": 0, "error_details": str(e)})

@router.websocket("/live")
async def live_endpoint(
    websocket: WebSocket,
    client_id: str = "anonymous",
    partial_albert: bool = False,
    device_id: str | None = None,
    session_id: str | None = None,
):
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_"):
        await websocket.close(code=4003)
        return
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
        # Toujours enregistrer un job si un session_id est fourni, même si l'audio est vide,
        # pour assurer la cohérence de l'interface et la stabilité des tests.
        if session_id:
            wav_path, timestamp = await session.save_wav_only()
            job_id = session_id
            
            if wav_path:
                add_job(job_id, {"status": "en_attente", "progress": 0})
                asyncio.create_task(_live_final_job(wav_path, job_id, client_id, timestamp))
            else:
                # Session vide mais identifiée : on marque comme terminé (vide)
                add_job(job_id, {"status": "terminé", "progress": 100, "result_file": None})