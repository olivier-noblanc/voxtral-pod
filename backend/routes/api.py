from __future__ import annotations
from fastapi import APIRouter
from backend.routes import (
    system,
    transcriptions,
    audio,
    postprocess,
    diarization,
    speakers,
    live,
    utils
)
from backend.state import get_asr_engine, add_job
from backend.routes.utils import _update_job_status

router = APIRouter()

# Include sub-routers
router.include_router(system.router)
router.include_router(transcriptions.router)
router.include_router(audio.router)
router.include_router(postprocess.router)
router.include_router(diarization.router)
router.include_router(speakers.router)
router.include_router(live.router)

# Expose utility functions at the router level for tests
_safe_join = utils._safe_join
# Expose functions that are imported in tests but not in sub-routers
_assemble_chunks = audio._assemble_chunks
update_segment_speaker = speakers.update_segment_speaker
# Import TRANSCRIPTIONS_DIR directly from config since it's not explicitly exported
from backend.config import TRANSCRIPTIONS_DIR
# Expose the internal function needed for testing
_extract_and_save_profile_sync = speakers._extract_and_save_profile_sync
# Expose functions needed for testing
get_asr_engine = get_asr_engine
add_job = add_job
_update_job_status = _update_job_status
# Expose _gpu_job for testing
_gpu_job = audio._gpu_job

# Add missing routes that were referenced in JS but not properly exposed
# These are needed for the frontend to work properly and tests to pass
from backend.routes.postprocess import (
    generate_summary,
    generate_actions,
    generate_cleanup,
    view_summary,
    view_actions,
    view_cleanup
)
from backend.routes.audio import batch_chunk_route, download_audio
from backend.routes.system import change_model_route, rate_limiter_status, git_status, git_update, trigger_cleanup
from backend.routes.speakers import update_segment_speaker as update_segment_speaker_route, get_speaker_profiles
from backend.routes.transcriptions import download_transcript, get_trans, list_trans, delete_trans, view_transcription
from backend.routes.utils import _safe_join
from backend.routes.system import status_route

# Expose the missing routes at the router level
router.post("/summary/{filename}")(generate_summary)
router.post("/actions/{filename}")(generate_actions)
router.post("/cleanup/{filename}")(generate_cleanup)
router.get("/view_summary/{client_id}/{filename}")(view_summary)
router.get("/view_actions/{client_id}/{filename}")(view_actions)
router.get("/view_cleanup/{client_id}/{filename}")(view_cleanup)
router.post("/batch_chunk")(batch_chunk_route)
router.post("/change_model")(change_model_route)
router.post("/segment_update")(update_segment_speaker_route)
router.get("/download_transcript/{client_id}/{filename}")(download_transcript)
router.get("/transcriptions/")(list_trans)
router.get("/transcriptions/{filename}")(get_trans)
router.delete("/transcriptions/{filename}")(delete_trans)
router.get("/view/{client_id}/{filename}")(view_transcription)
router.get("/download_audio/{client_id}/{filename}")(download_audio)
router.get("/rate_limiter_status")(rate_limiter_status)
router.get("/git_status")(git_status)
router.post("/git_update")(git_update)
router.post("/cleanup")(trigger_cleanup)
router.get("/speaker_profiles")(get_speaker_profiles)
router.get("/status/{file_id}")(status_route)

