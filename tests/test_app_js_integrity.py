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