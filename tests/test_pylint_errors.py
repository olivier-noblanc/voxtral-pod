import pathlib
import py_compile

def test_syntax():
    """
    Verify that the entire project compiles without syntax errors,
    ignoring the .history directory.
    """
    for file_path in pathlib.Path(".").rglob("*.py"):
        if ".history" in file_path.parts:
            continue
        py_compile.compile(str(file_path), doraise=True)
