#!/bin/bash

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
echo "[*] Syncing from GitHub (force)..."
git fetch origin main && git reset --hard origin/main || echo "[!] Local mode (sync failed)"

# 2. Virtual environment
FORCE_REINSTALL=false
if [ -d "$VENV_DIR" ]; then
    if ! "$VENV_DIR/bin/python" -c "import torch; import webrtcvad; import faster_whisper" &> /dev/null; then
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

# 3. Dependencies
echo "[*] Checking/Installing dependencies..."
pip install -U pip setuptools wheel
if ! pip install -r requirements.txt; then
    echo "[!] Retrying with webrtcvad from git..."
    pip install "git+https://github.com/wiseman/py-webrtcvad.git"
    pip install -r requirements.txt
fi

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
"$VENV_DIR/bin/python" server_asr.py