import re
from pathlib import Path
from typing import Set

# Files under test

APP_JS_PATH = Path(__file__).parents[1] / "static" / "app.js"
API_PY_PATH = Path(__file__).parents[1] / "backend" / "routes" / "api.py"


def _extract_fetch_paths(js_code: str) -> Set[str]:
    """
    Return a set of route paths used in ``fetch`` calls inside app.js.
    The pattern captures the literal string passed to ``fetch`` after the leading '/'.
    """
    pattern = re.compile(r"""fetch\(\s*['"]/(.+?)['"]""")
    return {m.group(1).split("?")[0] for m in pattern.finditer(js_code)}


def _extract_api_routes(api_code: str) -> Set[str]:
    """
    Return a set of route paths declared in the FastAPI router.
    Looks for ``@router.<method>("/path")`` decorators.
    """
    pattern = re.compile(r"""@router\.\w+\(\s*["']/(.+?)["']""")
    return {m.group(1) for m in pattern.finditer(api_code)}


def test_fetch_routes_have_backend_implementation() -> None:
    """
    For each ``fetch('/...')`` call found in static/app.js, ensure that a
    corresponding route is declared in backend/routes/api.py.
    """
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    api_code = API_PY_PATH.read_text(encoding="utf-8")

    fetch_paths = _extract_fetch_paths(js_code)
    api_routes = _extract_api_routes(api_code)

    missing = fetch_paths - api_routes
    assert not missing, f"The following routes are used in app.js but missing in api.py: {sorted(missing)}"


def test_api_routes_are_used_in_frontend() -> None:
    """
    Ensure that every route declared in the backend is referenced at least once
    in the frontend JavaScript. This guards against dead endpoints.
    """
    js_code = APP_JS_PATH.read_text(encoding="utf-8")
    api_code = API_PY_PATH.read_text(encoding="utf-8")

    fetch_paths = _extract_fetch_paths(js_code)
    api_routes = _extract_api_routes(api_code)

    # On autorise certains endpoints à ne pas être utilisés par le frontend JS
    # (ex: outils de benchmark, API batch pur, ou features en cours de développement)
    allowed_unused = {
        "diarize",
        "benchmark",
        "speakers/enroll",
        "speakers/identify",
        "diarization_data/{client_id}/{filename}",
        "download_rttm/{client_id}/{filename}",
        "view_diarization/{client_id}/{filename}"
    }
    
    unused = (api_routes - fetch_paths) - allowed_unused
    msg = (
        f"The following backend routes are not referenced in app.js "
        f"and not explicitly allowed: {sorted(unused)}"
    )
    assert not unused, msg
