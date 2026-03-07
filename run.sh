#!/bin/bash

# Voxtral Pod Launcher v2.5 (Modular)
PORT=8000
VENV_DIR="venv_asr"
MODEL="${1:-voxtral}"

# Auto-update if repo exists
if [ -d ".git" ]; then
    echo "[*] Mise à jour du code via Git..."
    git pull origin main || echo "[!] Impossible de pull, utilisation de la version locale."
fi

# Setup environnement (allégé car géré par server_asr.py et requirements.txt)
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -U pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

export ASR_MODEL="$MODEL"
echo "==== Lancement du serveur Voxtral Modulaire ===="
python server_asr.py