#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/olivier-noblanc/voxtral-pod.git"
VENV_DIR="venv_asr"
MODEL="${1:-whisper}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "===================================================="
echo "   LANCEUR VOXTRAL-POD v3.1"
echo "===================================================="

echo "[#] Python : $(python3 --version 2>&1)"

# ------------------------------------------------------------------------------
# 0. PRECONDITIONS (NO INSTALL HERE)
# ------------------------------------------------------------------------------
if ! command -v ffmpeg &> /dev/null; then
    echo "[FATAL] ffmpeg manquant. Il doit être installé dans l'image Docker."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[FATAL] venv absent ($VENV_DIR). Build Docker incorrect."
    exit 1
fi

# Activation venv
source "$VENV_DIR/bin/activate"

# ------------------------------------------------------------------------------
# 1. GPU DETECTION
# ------------------------------------------------------------------------------
USE_GPU=0
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    USE_GPU=1
    echo "[*] GPU détecté"
else
    echo "[*] Mode CPU"
fi

# ------------------------------------------------------------------------------
# 2. GIT SYNC (OPTIONNEL EN PROD)
# ------------------------------------------------------------------------------
git config --global --add safe.directory "$PWD"

if [ ! -d ".git" ]; then
    echo "[*] Init repo..."
    git init .
    git remote add origin "$REPO_URL"
fi

git remote set-url origin "$REPO_URL"

if [ "${SKIP_GIT_RESET:-false}" = "true" ]; then
    echo "[*] Mode local (pas de reset)"
else
    echo "[*] Sync Git..."
    git fetch origin main && git reset --hard origin/main || echo "[WARN] Git sync failed"
fi

# ------------------------------------------------------------------------------
# 3. TORCH VALIDATION (CRITIQUE)
# ------------------------------------------------------------------------------
python - <<EOF
import torch, sys

gpu_expected = ${USE_GPU}

if gpu_expected:
    assert torch.cuda.is_available(), "Torch CPU détecté alors qu'un GPU est présent"
    print("[OK] Torch GPU confirmé")
else:
    assert not torch.cuda.is_available(), "Torch GPU installé en mode CPU"
    print("[OK] Torch CPU confirmé")

print("[INFO] Torch version:", torch.__version__)
EOF

# ------------------------------------------------------------------------------
# 4. LAUNCH
# ------------------------------------------------------------------------------
export ASR_MODEL="$MODEL"

echo "===================================================="
echo "LAUNCH: Model=$MODEL"
echo "===================================================="

CERT_FILE=$(ls *.pem 2>/dev/null | grep -i cert | head -1 || true)
KEY_FILE=$(ls *.pem 2>/dev/null | grep -i key | head -1 || true)

if [ -z "$CERT_FILE" ]; then
    CERT_FILE=$(ls *.crt 2>/dev/null | head -1 || true)
    KEY_FILE=$(ls *.key 2>/dev/null | head -1 || true)
fi

if [ -n "${CERT_FILE:-}" ] && [ -n "${KEY_FILE:-}" ]; then
    echo "[*] HTTPS"
    exec "$VENV_DIR/bin/uvicorn" backend.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --ssl-certfile "$CERT_FILE" \
        --ssl-keyfile "$KEY_FILE"
else
    echo "[*] HTTP"
    exec "$VENV_DIR/bin/uvicorn" backend.main:app \
        --host 0.0.0.0 \
        --port 8000
fi