import pathlib
import re

APP_JS_PATH = pathlib.Path(__file__).parents[1] / "static" / "app.js"

def test_albert_buttons_use_window_open():
    """
    Ensure that Albert buttons (Summary, Actions, Clean) use window.open 
    and the correct view routes.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    
    # Check generateSummary
    assert re.search(r"function generateSummary\(.*\)\s*\{[\s\S]*window\.open\(\"/view_summary/\"", content), \
        "generateSummary does not use window.open to /view_summary/"
        
    # Check generateActions
    assert re.search(r"function generateActions\(.*\)\s*\{[\s\S]*window\.open\(\"/view_actions/\"", content), \
        "generateActions does not use window.open to /view_actions/"
        
    # Check cleanText
    assert re.search(r"function cleanText\(.*\)\s*\{[\s\S]*window\.open\(\"/view_cleanup/\"", content), \
        "cleanText does not use window.open to /view_cleanup/"

def test_no_formatTextForOS_leftover():
    """
    Ensure the formatTextForOS utility was actually removed from app.js as it's no longer needed.
    """
    content = APP_JS_PATH.read_text(encoding="utf-8")
    assert "function formatTextForOS" not in content, \
        "formatTextForOS utility was found in app.js - cleaning incomplete."
