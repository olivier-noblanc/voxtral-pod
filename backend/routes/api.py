from __future__ import annotations
from fastapi import APIRouter
from backend.routes import (
    system,
    transcriptions,
    audio,
    postprocess,
    diarization,
    speakers,
    live
)

router = APIRouter()

# Include sub-routers
router.include_router(system.router)
router.include_router(transcriptions.router)
router.include_router(audio.router)
router.include_router(postprocess.router)
router.include_router(diarization.router)
router.include_router(speakers.router)
router.include_router(live.router)
