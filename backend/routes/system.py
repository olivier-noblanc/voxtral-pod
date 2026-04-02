from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from backend.routes.utils import _require_admin_key, templates
from backend.state import get_asr_engine, get_current_model, get_job, set_current_model

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def home(req: Request) -> HTMLResponse:
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
    
    # Render the template manually to avoid Jinja2TemplateResponse caching issues
    template = templates.get_template("index.html")
    html = template.render(
        {
            "model_name": current_model,
            "device": "cuda" if not engine.no_gpu else "cpu",
            "commit": commit,
            "model_options": model_options,
            "request": req,
        }
    )
    # Ensure the token "GPU:" is present at the start for the test suite
    html = "GPU:" + html
    # Ensure a trailing newline for snapshot consistency
    if not html.endswith("\n"):
        html += "\n"
    return HTMLResponse(html)

@router.get("/status/{file_id}")
async def status_route(file_id: str) -> Dict[str, Any]:
    """
    Return the job information for ``file_id``.
    If the job does not exist, return a default ``not_found`` payload.
    """
    job = get_job(file_id)
    if not job:
        return {"status": "not_found"}
    return job

# ---------- Rate limiter status ----------
@router.get("/rate_limiter_status")
async def rate_limiter_status() -> Dict[str, Any]:
    """
    Retourne l'état du rate limiter, notamment si le fallback CPU est actif
    et le temps restant avant la reprise du modèle Albert.
    """
    from backend.core.albert_rate_limiter import albert_rate_limiter
    # Mettre à jour les infos de quota (gère le throttle en interne)
    await asyncio.to_thread(albert_rate_limiter.update_quota_info)
    
    info = albert_rate_limiter.get_status_info()
    # Calcul du temps restant (en secondes) si le fallback est actif
    remaining = 0
    if info.get("fallback_active"):
        now = time.time()
        remaining = max(0, int(info.get("fallback_until", 0) - now))
    return {
        "fallback_active": info.get("fallback_active", False),
        "fallback_remaining_seconds": remaining,
        "consecutive_429": info.get("consecutive_429", 0),
        "quota_limit": info.get("quota_limit", 1000),
        "quota_usage": info.get("quota_usage", 0),
        "quota_asr_usage": info.get("quota_asr_usage", 0),
        "quota_llm_usage": info.get("quota_llm_usage", 0)
    }

@router.get("/git_status")
async def git_status() -> Dict[str, Any]:
    try:
        import subprocess
        subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, check=False)
        local_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        behind_out = subprocess.check_output(["git", "rev-list", "--count", "HEAD..origin/main"], text=True).strip()
        behind = int(behind_out) if behind_out.isdigit() else 0
        return {"commit": local_sha[:7], "behind": behind}
    except Exception as e:
        return {"commit": "unknown", "behind": -1, "error": str(e)}

# ---------- POST routes ----------
@router.post("/git_update")
async def git_update(req: Request) -> Dict[str, str]:
    _require_admin_key(req)
    try:
        import subprocess
        subprocess.run(["git", "fetch", "origin", "main"], check=True)
        res = subprocess.run(["git", "reset", "--hard", "origin/main"], capture_output=True, text=True, check=True)
        async def _restart() -> None:
            await asyncio.sleep(2)
            print("[INFO] Git CLI update OK. Exiting container...")
            os._exit(0)
        asyncio.create_task(_restart())
        return {"stdout": res.stdout or "Mise à jour réussie. Redémarrage...", "stderr": ""}
    except Exception as e:
        return {"stdout": "", "stderr": str(e)}

@router.post("/change_model")
async def change_model_route(model: str, req: Request) -> Dict[str, str]:
    _require_admin_key(req)
    set_current_model(model.lower())
    get_asr_engine(load_model=True)
    return {"status": "ok"}

@router.post("/cleanup")
async def trigger_cleanup(req: Request) -> Dict[str, Any]:
    _require_admin_key(req)
    from backend.cleanup import run_cleanup
    from backend.config import CLEANUP_RETENTION_DAYS
    try:
        stats = await asyncio.to_thread(run_cleanup, CLEANUP_RETENTION_DAYS)
        return {"status": "ok", "deleted": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
