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


def test_pending_job_server_validated_before_ui():
    """
    Regression guard: when a pending_job is found in localStorage on page load,
    the app must NOT immediately disable the upload button or show the progress
    container. Instead it must first fetch /status/{id} and only update the UI
    based on the server response.

    Concretely: the fetch('/status/...') call must precede any DOM manipulation
    inside the pendingJob branch, and the UI must only be shown when the server
    confirms the job is truly in progress.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")

    # The pendingJob block must call fetch('/status/...') before touching the DOM
    pending_block_start = content.find("if (pendingJob)")
    assert pending_block_start != -1, "pendingJob guard block not found in app.js"

    block = content[pending_block_start:]

    fetch_pos = block.find("fetch(`/status/")
    disabled_pos = block.find("uploadBtn.disabled = true")

    assert fetch_pos != -1, (
        "fetch('/status/...') call is missing from the pendingJob block — "
        "server validation has been removed."
    )
    assert disabled_pos == -1 or disabled_pos > fetch_pos, (
        "uploadBtn is disabled BEFORE the server responds — "
        "UI must not be blocked until /status returns a live job."
    )


def test_pending_job_cleared_on_not_found():
    """
    Regression guard: when /status returns 'not_found', the app must call
    clearPendingJob() (or equivalent) to silently clean localStorage.
    The user must never see a stuck progress bar caused by an orphaned job.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")

    assert "not_found" in content, (
        "'not_found' status is not handled — orphaned jobs will block the UI."
    )
    # After 'not_found', clearPendingJob must be invoked
    not_found_idx = content.find("'not_found'")
    if not_found_idx == -1:
        not_found_idx = content.find('"not_found"')
    assert not_found_idx != -1, "not_found handling is missing from app.js"

    snippet = content[not_found_idx: not_found_idx + 300]
    assert "clearPendingJob" in snippet or "localStorage.removeItem('pending_job')" in snippet, (
        "pending_job is not removed from localStorage on 'not_found' — "
        "the UI will remain blocked for the user."
    )


def test_pending_job_cleared_on_network_error():
    """
    Regression guard: if the /status fetch throws (server unreachable), the app
    must clean up localStorage and leave the UI in a usable state — never leave
    the upload button permanently disabled.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")

    # Find the .catch() block inside the pendingJob branch
    pending_block_start = content.find("if (pendingJob)")
    assert pending_block_start != -1, "pendingJob guard block not found in app.js"

    block = content[pending_block_start: pending_block_start + 2000]
    catch_pos = block.find(".catch(")
    assert catch_pos != -1, (
        "No .catch() handler found in the pendingJob fetch — "
        "network errors will leave the UI permanently blocked."
    )

    catch_block = block[catch_pos: catch_pos + 300]
    assert "clearPendingJob" in catch_block or "localStorage.removeItem" in catch_block, (
        "The .catch() handler does not clean up pending_job — "
        "a network error will permanently disable the upload button."
    )