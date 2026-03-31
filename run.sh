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

# 0.1. System dependencies (ffmpeg + codecs)
if command -v apt-get &> /dev/null; then
    echo "[*] Ensuring ffmpeg and extra codecs are installed..."
    apt-get update && apt-get install -y ffmpeg libavcodec-extra || true
elif ! command -v ffmpeg &> /dev/null; then
    echo "[*] ffmpeg missing. Attempting installation..."
    if command -v yum &> /dev/null; then
         yum install -y ffmpeg
    else
        echo "[!] Auto-install failed. Please install ffmpeg manually."
    fi
fi

USE_GPU=0
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    USE_GPU=1
fi

# 1. Initialisation du dépôt git local si nécessaire
git config --global --add safe.directory "$PWD"
if [ ! -d ".git" ]; then
    echo "[*] Init code..."
    git init .
    git remote add origin "$REPO_URL"
fi
git remote set-url origin "$REPO_URL"

# 2. Sync from Remote Repo
if [ "$SKIP_GIT_RESET" = "true" ]; then
    echo "[*] Local modifications preserved (SKIP_GIT_RESET=true)."
else
    echo "[*] Syncing from GitHub (force update)..."
    git fetch origin main && git reset --hard origin/main || echo "[!] Sync failed, local mode enabled."
fi




# 5. Launch
export ASR_MODEL="$MODEL"
echo "===================================================="
echo "🚀 LAUNCH: Model=$MODEL, Language=FR"
echo "===================================================="


# === DETECTION CERTIFICAT ===
CERT_FILE=$(ls *.pem 2>/dev/null | grep -i "cert" | head -1)
KEY_FILE=$(ls *.pem 2>/dev/null | grep -i "key" | head -1)

# Fallback sur .crt si pas de .pem
if [ -z "$CERT_FILE" ]; then
    CERT_FILE=$(ls *.crt 2>/dev/null | head -1)
    KEY_FILE=$(ls *.key 2>/dev/null | head -1)
fi

python - <<EOF
import torch
assert not torch.cuda.is_available(), "Torch GPU installé en mode CPU !"
print("[OK] Torch CPU confirmé")
EOF

if [ -n "$CERT_FILE" ] && [ -n "$KEY_FILE" ]; then
    echo "[*] Certificat détecté : $CERT_FILE"
    echo "[*] Clé détectée       : $KEY_FILE"
    echo "[*] Démarrage en HTTPS → https://10.25.22.104:8000"
    "$VENV_DIR/bin/uvicorn" backend.main:app  --host 0.0.0.0 --port 8000 --ssl-certfile "$CERT_FILE"  --ssl-keyfile  "$KEY_FILE"
else
    echo "[!] Aucun certificat trouvé → démarrage en HTTP"
    echo "[*] Démarrage en HTTP  → http://10.25.22.104:8084"
    
    "$VENV_DIR/bin/uvicorn" backend.main:app  --host 0.0.0.0  --port 8000
fi
