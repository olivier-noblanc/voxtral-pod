from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import re
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates

from backend.config import BASE_DIR
from backend.state import update_job

logger = logging.getLogger(__name__)

# Markdown handling: use real markdown if available, otherwise fallback
_markdown_spec = importlib.util.find_spec("markdown")
if _markdown_spec is not None:
    import markdown as _markdown_lib
    markdown = _markdown_lib
else:
    class _FallbackMarkdown:
        @staticmethod
        def markdown(text: str, **kwargs: object) -> str:
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

# Setup Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "backend", "templates"))
templates.env.cache = None  # Disable Jinja2 caching to avoid unhashable globals issue

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

def _validate_client_id(client_id: str) -> None:
    """Vérifie que le client_id est valide (format user_xxxxxxxx)."""
    if not client_id or client_id == "anonymous" or not client_id.startswith("user_") or len(client_id) < 6:
        raise HTTPException(status_code=400, detail="Identifiant client (UUID) invalide ou anonyme non autorisé.")

def _require_admin_key(request: Request) -> None:
    admin_key = os.getenv("ADMIN_API_KEY")
    if not admin_key:
        return
    provided = request.headers.get("X-Admin-Key", "")
    if provided != admin_key:
        raise HTTPException(status_code=403, detail="Clé admin invalide ou manquante.")

def _update_job_status(job_id: str, status: str, progress: int = 0, result_file: Optional[str] = None) -> None:
    data = {"status": status, "progress": progress}
    if result_file:
        data["result_file"] = result_file
    update_job(job_id, data)
