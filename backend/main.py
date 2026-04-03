from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.config import BASE_DIR, CLEANUP_RETENTION_DAYS
from backend.routes import api as api_module
from backend.state import get_asr_engine

# Silence noisy loggers
for noisy_logger in ["multipart.multipart", "faster_whisper", "resemble_enhance"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Import cleanup task
from backend.cleanup import periodic_cleanup_task


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # startup
    from backend.state import init_db

    init_db()
    get_asr_engine(load_model=False)

    cleanup_task = asyncio.create_task(periodic_cleanup_task(CLEANUP_RETENTION_DAYS))
    yield
    # shutdown
    cleanup_task.cancel()


app = FastAPI(title="SOTA ASR Server", version="4.0.0", lifespan=lifespan)

# Trusted proxies for Real-IP
_TRUSTED_PROXIES = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",")
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=_TRUSTED_PROXIES)  # type: ignore[arg-type]

# Static files should be outside backend/ in the repo root/static or build/
static_path = os.path.join(BASE_DIR, "static")
if not os.path.isdir(static_path):
    os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Include routers
app.include_router(api_module.router)


# Minimal SSR middleware for HTML response if needed (optional)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    import time

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
