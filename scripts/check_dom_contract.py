#!/usr/bin/env python3
"""
check_dom_contract.py

Pre-commit hook that verifies the DOM contract between the JavaScript front-end
(`static/app.js`) and the HTML UI defined in `backend/html_ui.py`.

Exit codes:
  * 0 – No problems.
  * 1 – Missing IDs or detected anti-patterns.
"""

import re
import sys
from pathlib import Path


def extract_js_ids(js_code: str) -> set[str]:
    """Return a set of IDs referenced in the JavaScript source."""
    ids = set()
    ids.update(re.findall(r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""", js_code))
    ids.update(re.findall(r"""querySelector\(\s*['"]#([^'"]+)['"]\s*\)""", js_code))
    return ids


def extract_html_ids(html_code: str) -> set[str]:
    """Return a set of IDs present in the HTML string."""
    return set(re.findall(r"""id\s*=\s*['"]([^'"]+)['"]""", html_code))


def find_add_event_listener_issues(js_code: str) -> list[str]:
    """
    Detect addEventListener calls inside an if(el) block without console log.
    Returns variable names matching the anti-pattern.
    """
    issues = []
    var_assignments = re.findall(
        r"""(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*getElementById\(\s*['"][^'"]+['"]\s*\)""",
        js_code,
    )
    for var in var_assignments:
        pattern = rf"""if\s*\(\s*{re.escape(var)}\s*\)\s*\{{([^{{}}]*?addEventListener[^\{{}}]*?)\}}"""
        match = re.search(pattern, js_code, re.DOTALL)
        if match:
            block = match.group(1)
            if not re.search(r"""console\.\w+""", block):
                issues.append(var)
    return issues


def find_ws_send_issues(js_code: str) -> list[int]:
    """
    Detect ws.send() calls not guarded by a readyState check.
    Returns line numbers where the issue occurs.
    """
    issues = []
    lines = js_code.splitlines()
    for i, line in enumerate(lines, start=1):
        if re.search(r"""\bws\.send\(""", line):
            guard_found = any(
                re.search(r"""if\s*\(\s*ws\.readyState\s*===\s*1\s*\)""", lines[j])
                for j in range(max(0, i - 5), i)
            )
            if not guard_found:
                issues.append(i)
    return issues


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    js_path   = project_root / "static"  / "app.js"
    html_path = project_root / "backend" / "html_ui.py"

    if not js_path.is_file():
        print(f"❌ JS file not found: {js_path}", file=sys.stderr)
        return 1
    if not html_path.is_file():
        print(f"❌ HTML file not found: {html_path}", file=sys.stderr)
        return 1

    js_code   = js_path.read_text(encoding="utf-8")
    html_code = html_path.read_text(encoding="utf-8")

    missing_ids      = extract_js_ids(js_code) - extract_html_ids(html_code)
    listener_issues  = find_add_event_listener_issues(js_code)
    ws_issues        = find_ws_send_issues(js_code)

    if missing_ids:
        print("❌ IDs in JS but missing in HTML:")
        for mid in sorted(missing_ids):
            print(f"   - {mid}")

    if listener_issues:
        print("\n⚠️  addEventListener inside if(el) without console log:")
        for var in listener_issues:
            print(f"   - {var}")

    if ws_issues:
        print("\n❌ ws.send() without readyState guard on lines:")
        for ln in ws_issues:
            print(f"   - line {ln}")

    if missing_ids or listener_issues or ws_issues:
        return 1

    print("✅ DOM contract OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())