import re
from pathlib import Path
from typing import Set

APP_JS_PATH = Path(__file__).parents[1] / "static" / "app.js"
ROUTES_DIR = Path(__file__).parents[1] / "backend" / "routes"

ROUTE_FILES = [
    ROUTES_DIR / "api.py",
    ROUTES_DIR / "audio.py",
    ROUTES_DIR / "diarization.py",
    ROUTES_DIR / "live.py",
    ROUTES_DIR / "postprocess.py",
    ROUTES_DIR / "speakers.py",
    ROUTES_DIR / "system.py",
    ROUTES_DIR / "transcriptions.py",
]


def _extract_fetch_paths(js_code: str) -> Set[str]:
    """Routes appelées via fetch('/...') littéral dans app.js."""
    pattern = re.compile(r"""fetch\(\s*['"]/(.+?)['"]""")
    return {m.group(1).split("?")[0] for m in pattern.finditer(js_code)}


def _extract_api_routes(api_code: str) -> Set[str]:
    """Routes déclarées via @router.<method>("/path")."""
    pattern = re.compile(r"""@router\.\w+\(\s*["']/(.+?)["']""")
    return {m.group(1) for m in pattern.finditer(api_code)}


def _all_api_routes() -> Set[str]:
    """Agrège les routes de tous les fichiers du dossier routes/."""
    routes: Set[str] = set()
    for path in ROUTE_FILES:
        if path.exists():
            routes |= _extract_api_routes(path.read_text(encoding="utf-8"))
    return routes


def test_fetch_routes_have_backend_implementation() -> None:
    """Pour chaque fetch('/...') dans app.js, vérifier qu'une route backend existe."""
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    fetch_paths = _extract_fetch_paths(js_code)
    api_routes = _all_api_routes()

    missing = fetch_paths - api_routes
    assert not missing, (
        f"The following routes are used in app.js but missing in backend routes: {sorted(missing)}"
    )


def test_api_routes_are_used_in_frontend() -> None:
    """Vérifier que chaque route backend est référencée dans le frontend."""
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    fetch_paths = _extract_fetch_paths(js_code)
    api_routes = _all_api_routes()

    # Routes non détectables par regex de string littérale :
    # - Routes à paramètres dynamiques (construites par concaténation JS)
    # - Routes WebSocket (new WebSocket(), pas fetch())
    # - Endpoints internes non exposés au frontend
    allowed_unused = {
        "diarize",
        "benchmark",
        "speakers/enroll",
        "speakers/identify",
        "view/{client_id}/{filename}",
        "view_summary/{client_id}/{filename}",
        "view_actions/{client_id}/{filename}",
        "view_cleanup/{client_id}/{filename}",
        "download_audio/{client_id}/{filename}",
        "download_transcript/{client_id}/{filename}",
        "diarization_data/{client_id}/{filename}",
        "download_rttm/{client_id}/{filename}",
        "view_diarization/{client_id}/{filename}",
        "status/{file_id}",
        "transcriptions/{filename}",
        "transcription/{filename}",
        "actions/{filename}",
        "cleanup/{filename}",
        "summary/{filename}",
        "live",
        "transcriptions",
        "transcriptions/",
        "speaker_profiles",
        "cleanup",
        "job_log/{job_id}",
    }

    unused = (api_routes - fetch_paths) - allowed_unused
    assert not unused, (
        f"The following backend routes are not referenced in app.js "
        f"and not explicitly allowed: {sorted(unused)}"
    )
