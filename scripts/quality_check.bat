@echo off
REM Quality Check Pipeline for voxtral-pod (Windows)

echo [1/4] Clearing Pytest Cache...
pytest --cache-clear -q

echo [2/4] Running Ruff Linter...
ruff check . --fix

echo [3/4] Running Mypy Strict Type Check...
mypy --strict .

echo [4/4] Running Pytest Suite with Coverage...
pytest --cov=backend --cov-report=term-missing --cov-report=html

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline failed.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] All quality checks passed!
