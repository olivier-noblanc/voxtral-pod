#!/bin/bash

# ==============================================================================
# Voxtral Pod Launcher v6.0 (THE GIT COMPILATION EDITION)
# ==============================================================================

REPO_URL="https://github.com/olivier-noblanc/voxtral-pod.git"
VENV_DIR="venv_asr"
MODEL="${1:-voxtral}"

echo "===================================================="
echo "   🎙️  LANCEUR VOXTRAL-POD v14.0 (AUTO-REPAIR) 🎙️"
echo "===================================================="

# Diagnostic version
echo "[#] Python Système : $(python3 --version 2>&1)"

# 1. Récupération & Réparation du code
git config --global --add safe.directory "$PWD"
if [ ! -d ".git" ]; then
    echo "[*] Initialisation du code..."
    git init .
    git remote add origin "$REPO_URL"
fi
git remote set-url origin "$REPO_URL"
echo "[*] Synchronisation GitHub (Force)..."
git fetch origin main && git reset --hard origin/main || echo "[!] Mode local (échec sync)"

# 2. Gestion de l'environnement virtuel (Check agressif)
FORCE_REINSTALL=false
if [ -d "$VENV_DIR" ]; then
    # Test critique d'import incluant webrtcvad (test du bug pkg_resources)
    if ! "$VENV_DIR/bin/python" -c "import torch; import webrtcvad" &> /dev/null; then
        echo "[!] Venv incomplet ou incompatible (Torch ou WebRTC plantent). Recréation..."
        rm -rf "$VENV_DIR"
        FORCE_REINSTALL=true
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Création du venv avec accès système..."
    PYTHON_CMD="python3"
    command -v python3.11 &> /dev/null && PYTHON_CMD="python3.11"
    $PYTHON_CMD -m venv --system-site-packages "$VENV_DIR"
    FORCE_REINSTALL=true
fi

source "$VENV_DIR/bin/activate"

# 3. Installation des dépendances
echo "[*] Optimisation des dépendances..."
pip install -U pip setuptools wheel

if [ "$FORCE_REINSTALL" = true ] || ! pip show fastapi &> /dev/null; then
    echo "[*] Installation via requirements.txt..."
    # Si requirements.txt échoue, on tente une compilation directe de webrtcvad depuis Git
    if ! pip install -r requirements.txt; then
        echo "[!] Echec de pip install -r. Tentative de compilation directe de webrtcvad..."
        pip install "git+https://github.com/wiseman/py-webrtcvad.git"
        pip install -r requirements.txt
    fi
else
    echo "[*] Dépendances OK."
fi

# 4. Lancement
export ASR_MODEL="$MODEL"
echo "===================================================="
echo "🚀 LANCEMENT : Modèle=$MODEL"
echo "===================================================="
"$VENV_DIR/bin/python" server_asr.py