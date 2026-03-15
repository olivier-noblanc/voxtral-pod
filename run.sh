#!/usr/bin/env bash

# ==============================================================================
# Voxtral Pod Launcher v3.0 (TranscriptionSuite-compatible)
# ==============================================================================

REPO_URL="https://github.com/olivier-noblanc/voxtral-pod.git"
VENV_DIR="venv_asr"
MODEL="${1:-whisper}"

# Reduce VRAM fragmentation (recommended by PyTorch for OOM prevention)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "===================================================="
echo "   🎙️  LANCEUR VOXTRAL-POD v3.0 (FR)       🎙️"
echo "===================================================="

# Diagnostic
echo "[#] Python Système : $(python3 --version 2>&1)"

# 0.1. System dependencies (ffmpeg)
if ! command -v ffmpeg &> /dev/null; then
    echo "[*] ffmpeg missing. Attempting installation..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y ffmpeg || apt-get update && apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    else
        echo "[!] Auto-install failed. Please install ffmpeg manually."
    fi
fi


# 1. Code sync
git config --global --add safe.directory "$PWD"
if [ ! -d ".git" ]; then
    echo "[*] Init code..."
    git init .
    git remote add origin "$REPO_URL"
fi
git remote set-url origin "$REPO_URL"

# 1. Sync from Remote Repo
if [ "$SKIP_GIT_RESET" = "true" ]; then
    echo "[*] Local modifications preserved (SKIP_GIT_RESET=true)."
else
    echo "[*] Syncing from GitHub (force update)..."
    git fetch origin main && git reset --hard origin/main || echo "[!] Sync failed, local mode enabled."
fi

# 2. Virtual environment
FORCE_REINSTALL=false

# Check if requirements.txt has changed
REQ_HASH_FILE="$VENV_DIR/req_hash.txt"
CURRENT_HASH=$(md5sum requirements.txt | cut -d' ' -f1)

if [ -d "$VENV_DIR" ]; then
    if [ ! -f "$REQ_HASH_FILE" ] || [ "$(cat "$REQ_HASH_FILE")" != "$CURRENT_HASH" ]; then
        echo "[*] Requirements changed. Checking for updates..."
        # We don't delete, pip will handle incremental updates
        FORCE_REINSTALL=true
    elif ! "$VENV_DIR/bin/python" -c "import torch; import webrtcvad; import faster_whisper" &> /dev/null; then
        echo "[!] Venv incomplete or broken. Recreating..."
        rm -rf "$VENV_DIR"
        FORCE_REINSTALL=true
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating venv..."
    PYTHON_CMD="python3"
    command -v python3.11 &> /dev/null && PYTHON_CMD="python3.11"
    $PYTHON_CMD -m venv "$VENV_DIR"
    FORCE_REINSTALL=true
fi

source "$VENV_DIR/bin/activate"
# Store hash after activation to mark as compliant
echo "$CURRENT_HASH" > "$REQ_HASH_FILE"

# 3. Dependencies
echo "[*] Starting robust dependency check..."

"$VENV_DIR/bin/python" - <<EOF
import subprocess
import sys
import os

MAPPING = {
    "uvicorn[standard]": "uvicorn",
    "python-multipart": "multipart",
    "faster-whisper": "faster_whisper",
    "mistral-common[audio]": "mistral_common",
    "ffmpeg-python": "ffmpeg",
    "webrtcvad-wheels": "webrtcvad",
    "silero-vad": "silero_vad"
}

req_file = "requirements.txt"
if not os.path.exists(req_file):
    print(f"[!] {req_file} not found.")
    sys.exit(1)

with open(req_file, "r") as f:
    lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

print(f"[*] Found {len(lines)} dependencies to check.")

for line in lines:
    base_pkg = line.split(">=")[0].split("==")[0].split("<=")[0].strip()
    import_name = MAPPING.get(base_pkg, base_pkg.split("[")[0].replace("-", "_"))
    
    # Debug: print what we are doing
    sys.stdout.write(f"  [WAIT] {base_pkg}... ")
    sys.stdout.flush()

    try:
        root_module = import_name.split(".")[0]
        __import__(root_module)
        print("\r  [OK]  " + base_pkg + " " * 10)
    except ImportError:
        print("\r  [FIX] " + base_pkg + " (missing). Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", line])

EOF

# 4. GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "==== GPU Status ===="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# 5. Launch
export ASR_MODEL="$MODEL"
echo "===================================================="
echo "🚀 LAUNCH: Model=$MODEL, Language=FR"
echo "===================================================="
"$VENV_DIR/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000