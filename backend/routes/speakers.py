from __future__ import annotations

import json
import os
import pathlib
import re
from typing import Any, Dict, List, cast

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from backend.routes.transcriptions import _get_trans_dir

# from backend.config import TRANSCRIPTIONS_DIR
from backend.routes.utils import _safe_join

router = APIRouter()

@router.get("/speaker_profiles")
async def get_speaker_profiles() -> List[Dict[str, Any]]:
    import backend.core.speaker_profiles as speaker_profiles
    profiles = speaker_profiles.load_profiles()
    return [{"speaker_id": sid, "name": name} for sid, (name, _) in profiles.items()]

@router.post("/speaker_profiles")
async def upsert_speaker_profile(payload: Dict[str, Any]) -> Dict[str, str]:
    import backend.core.speaker_profiles as speaker_profiles
    speaker_id = payload.get("speaker_id")
    name = payload.get("name")
    embedding = payload.get("embedding")
    if not speaker_id or not name:
        raise HTTPException(status_code=400, detail="speaker_id and name are required.")
    emb_array = None
    if embedding is not None:
        try:
            emb_array = np.array(embedding, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid embedding format.")
    speaker_profiles.save_profile(speaker_id, name, emb_array)
    return {"status": "ok"}

def _extract_and_save_profile_sync(audio_path: str, start_s: float, end_s: float, speaker_id: str) -> None:
    if not os.path.isfile(audio_path):
        print(f"[!] Audio file {audio_path} not found for profile extraction.")
        return
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
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
        speaker_profiles.save_profile(speaker_id=speaker_id, name=speaker_id, embedding=cast(np.ndarray, emb))
        print(f"[*] Saved voice profile for speaker: {speaker_id}")
    except Exception as e:
        print(f"[!] Warning: Failed to extract speaker embedding for {speaker_id}: {e}")

@router.post("/segment_update")
async def update_segment_speaker(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, str]:
    client_id = payload.get("client_id")
    filename = payload.get("filename")
    segment_index = payload.get("segment_index")
    new_speaker = payload.get("new_speaker")
    # Explicitly check for None to satisfy Pyright
    if client_id is None or filename is None or new_speaker is None or not isinstance(segment_index, int):
        raise HTTPException(status_code=400, detail="Invalid payload: missing or invalid fields.")
    
    base_name = filename.replace(".json", "").replace(".txt", "")
    json_path = _safe_join(_get_trans_dir(), client_id, base_name + ".json")
    txt_path = _safe_join(_get_trans_dir(), client_id, base_name + ".txt")
    
    # 1. Priorité au JSON
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        if segment_index < 0 or segment_index >= len(segments):
            raise HTTPException(status_code=400, detail="segment_index out of range.")
        
        # Mise à jour du locuteur dans le JSON
        seg = segments[segment_index]
        seg["speaker"] = new_speaker
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
            
        # On tente de mettre à jour le RTTM aussi si présent
        rttm_path = _safe_join(_get_trans_dir(), client_id, base_name + ".rttm")
        if os.path.isfile(rttm_path):
            from backend.core.export import export_rttm
            export_rttm(segments, base_name, rttm_path)
            
        # On tente de mettre à jour le .diar.json aussi
        diar_path = _safe_join(_get_trans_dir(), client_id, base_name + ".diar.json")
        if os.path.isfile(diar_path):
            with open(diar_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2)

        # On déclenche l'enrôlement en arrière-plan
        start_s = float(seg.get("start", 0))
        end_s = float(seg.get("end", start_s + 1))
        if base_name.startswith("batch_"):
            ts = base_name.replace("batch_", "")
            audio_name = f"batch_{client_id}_{ts}.wav"
            audio_dir = "batch_audio"
        else:
            ts = base_name.replace("live_", "")
            audio_name = f"live_{client_id}_{ts}.wav"
            audio_dir = "live_audio"
        audio_path = _safe_join(_get_trans_dir(), audio_dir, audio_name)
        background_tasks.add_task(_extract_and_save_profile_sync, audio_path, start_s, end_s, new_speaker)
        
        return {"status": "ok"}

    # 2. Fallback legacy .txt (Regex)
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if segment_index < 0 or segment_index >= len(lines):
            raise HTTPException(status_code=400, detail="segment_index out of range.")
        
        match = re.search(r'\[([\d.]+)s\s*->\s*([\d.]+)s\]', lines[segment_index])
        if match:
            start_s = float(match.group(1))
            end_s = float(match.group(2))
            if base_name.startswith("batch_"):
                ts = base_name.replace("batch_", "")
                audio_name = f"batch_{client_id}_{ts}.wav"
                audio_dir = "batch_audio"
            else:
                ts = base_name.replace("live_", "")
                audio_name = f"live_{client_id}_{ts}.wav"
                audio_dir = "live_audio"
            audio_path = _safe_join(_get_trans_dir(), audio_dir, audio_name)
            background_tasks.add_task(_extract_and_save_profile_sync, audio_path, start_s, end_s, new_speaker)
        
        new_line = re.sub(
            r'(\[[^\]]*s\s*->\s*[^\]]*s\]\s*)\[[^\]]*\]',
            rf'\1[{new_speaker}]',
            lines[segment_index]
        )
        if new_line != lines[segment_index]:
            lines[segment_index] = new_line
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return {"status": "ok"}
    
    raise HTTPException(status_code=404, detail="Transcription (JSON or TXT) not found.")

@router.post("/speakers/enroll")
async def api_enroll_speaker(name: str = Form(...), file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Enrôle un nouveau locuteur (signature vocale).
    """
    import tempfile

    from backend.core.speaker_manager import SpeakerManager
    
    suffix = pathlib.Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        manager = SpeakerManager()
        success = manager.enroll_speaker(name, tmp_path)
        if success:
            return {"status": "ok", "message": f"Locuteur {name} enrôlé avec succès."}
        raise HTTPException(status_code=500, detail="Échec de l'enrôlement.")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/speakers/identify")
async def api_identify_speakers(file: UploadFile = File(...)) -> Any:
    """
    Identifie les locuteurs connus dans un fichier audio segmenté par diarisation.
    """
    import tempfile

    from backend.core.diarization_cpu import LightDiarizationEngine
    from backend.core.speaker_manager import SpeakerManager
    
    suffix = pathlib.Path(file.filename or "audio.wav").suffix
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
        return manager.identify_speakers_in_segments(tmp_path, segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
