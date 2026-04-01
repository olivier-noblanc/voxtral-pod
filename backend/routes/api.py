from __future__ import annotations

from fastapi import APIRouter

from backend import config as _config
from backend.routes import (
    audio,
    diarization,
    live,
    postprocess,
    speakers,
    system,
    transcriptions,
)
from backend.core.live import LiveSession
from backend.routes.utils import _safe_join as _safe_join_util
from backend.routes.utils import _update_job_status as _update_job_status_util
from backend.state import add_job as add_job_state
from backend.state import get_asr_engine as get_asr_engine_state

# ----------------------------------------------------------------------
# Explicit re-exports for Mypy --strict
# ----------------------------------------------------------------------
add_job = add_job_state
get_asr_engine = get_asr_engine_state
_update_job_status = _update_job_status_util
_safe_join = _safe_join_util
TRANSCRIPTIONS_DIR = _config.TRANSCRIPTIONS_DIR
_gpu_job = audio._gpu_job
_assemble_chunks = audio._assemble_chunks
update_segment_speaker = speakers.update_segment_speaker

__all__ = [
    "add_job",
    "get_asr_engine",
    "LiveSession",
    "_update_job_status",
    "_safe_join",
    "TRANSCRIPTIONS_DIR",
    "router",
    "_assemble_chunks",
    "update_segment_speaker",
]

router = APIRouter()

# ----------------------------------------------------------------------
# Include sub-routers — CE SONT EUX QUI DÉFINISSENT LES VRAIES ROUTES
# L'ordre est important : FastAPI prend le premier match.
# ----------------------------------------------------------------------
router.include_router(system.router)
router.include_router(transcriptions.router)
router.include_router(audio.router)
router.include_router(postprocess.router)
router.include_router(diarization.router)
router.include_router(speakers.router)
router.include_router(live.router)

# ----------------------------------------------------------------------
# Internal functions needed for tests
# ----------------------------------------------------------------------
_extract_and_save_profile_sync = speakers._extract_and_save_profile_sync

# ----------------------------------------------------------------------
# Imports for re-export (ne créent PAS de routes supplémentaires)
# ----------------------------------------------------------------------
from backend.routes.audio import batch_chunk_route, download_audio
from backend.routes.live import live_endpoint
from backend.routes.postprocess import (
    generate_actions,
    generate_cleanup,
    generate_summary,
    view_actions,
    view_cleanup,
    view_summary,
)
from backend.routes.speakers import get_speaker_profiles
from backend.routes.system import (
    change_model_route,
    git_status,
    rate_limiter_status,
    status_route,
)
from backend.routes.transcriptions import get_trans, list_trans, view_transcription
