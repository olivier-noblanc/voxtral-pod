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
