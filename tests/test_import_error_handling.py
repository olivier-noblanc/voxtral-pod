import pathlib
import re


def test_no_except_importerror_blocks():
    """
    Ensure that the codebase does not contain generic ``except ImportError`` blocks
    that silently mask missing dependencies. Such blocks can hide real errors
    (e.g., the fallback markdown renderer in ``backend/routes/api.py``).
    If a specific ``ImportError`` handling is required, it should be explicit
    and accompanied by a clear comment or fallback implementation.
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"except\s+ImportError")
    offending_files = []

    for py_file in project_root.rglob("*.py"):
        # Skip test files themselves
        if py_file.parts[-2] == "tests":
            continue
        # Skip history directory
        if ".history" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            # If a file cannot be read, skip it – it will be caught by other tests
            continue
        if pattern.search(content):
            offending_files.append(str(py_file.relative_to(project_root)))

    assert not offending_files, (
        "Found generic ``except ImportError`` blocks in the following files:\n"
        + "\n".join(offending_files)
        + "\n\nConsider handling the import error explicitly or adding a clear comment."
    )