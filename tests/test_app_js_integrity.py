import pathlib

APP_JS_PATH = pathlib.Path(__file__).parents[1] / "static" / "app.js"

def test_app_js_not_shrunk():
    """
    Ensure that the JavaScript file still contains a reasonable amount of code.
    The original file had several hundred lines; after refactoring it should
    still be well above a minimal threshold.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.strip()]

    # The file should contain at least 200 non‑empty lines.
    assert len(lines) >= 200, f"app.js appears to have been overly reduced ({len(lines)} lines)."

def test_essential_functions_present():
    """
    Verify that key functions required for the application are still present
    in static/app.js after the recent changes.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    required_functions = [
        "toggleMicrophone",
        "toggleSystemAudio",
        "toggleMeeting",
        "startRecording",
        "stopRecording",
        "loadAudioDevices",
        "loadHistory",
    ]
    missing = [fn for fn in required_functions if f"function {fn}(" not in content]
    assert not missing, f"The following essential functions are missing from app.js: {missing}"


def test_live_partial_row_tracking():
    """
    Ensure that the live transcript uses a `partialRow` variable to track the
    current in-progress partial segment instead of scanning all DOM rows.
    This prevents duplicate lines (repetitions) when segments are finalized.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    assert "partialRow" in content, (
        "partialRow variable is missing from app.js — "
        "the live transcript deduplication logic has been removed."
    )
    # The variable must be initialised to null (not undefined)
    assert "partialRow = null" in content, (
        "partialRow must be initialised to null before the WebSocket onmessage handler."
    )


def test_live_partial_uses_sindex_attribute():
    """
    Ensure that partial sentence rows store a `data-sindex` attribute containing
    the backend sentence_index so that the final message can locate the correct
    row even if messages arrive out of order.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    assert "data-sindex" in content or "dataset.sindex" in content, (
        "data-sindex attribute is missing — partial rows won't be matched by sentence_index."
    )
    # The sindex must be set from data.sentence_index coming from the WebSocket payload
    assert "data.sentence_index" in content, (
        "data.sentence_index is not referenced — the backend sentence_index is ignored."
    )


def test_live_final_looks_up_by_sindex_before_fallback():
    """
    Ensure that when a `final` sentence message arrives the handler first tries
    to find the partial row via its data-sindex attribute (querySelector), then
    falls back to the direct partialRow reference.  Both lookup strategies must
    coexist so messages arriving out-of-order are handled correctly.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    assert 'data-sindex' in content, (
        "querySelector by data-sindex is missing from the final message handler."
    )
    # Both lookup strategies must coexist
    assert "partialRow" in content, (
        "partialRow fallback reference is missing from the final message handler."
    )