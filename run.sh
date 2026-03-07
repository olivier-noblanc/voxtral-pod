#!/bin/bash

# Voxtral Pod Launcher v2.6 (Modular & Robust)
PORT=8000
VENV_DIR="venv_asr"
MODEL="${1:-voxtral}"

echo "[*] Démarrage du launcher Voxtral..."

# Auto-update if repo exists
if [ -d ".git" ]; then
    echo "[*] Vérification des mises à jour Git..."
    git pull origin main || echo "[!] Attention : impossible de pull les dernières modifications."
fi

# Choix de l'exécutable python (priorité 3.11)
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
fi

# Setup environnement
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Création de l'environnement virtuel avec $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Vérification de la présence de torch dans le venv
echo "[*] Vérification des dépendances..."
if ! python -c "import torch" &> /dev/null; then
    echo "[!] Torch non détecté ou venv corrompu. Installation des dépendances via requirements.txt..."
    pip install -U pip
    pip install -r requirements.txt
else
    echo "[*] Dépendances OK."
fi

export ASR_MODEL="$MODEL"
echo "==== Lancement du serveur Voxtral Modulaire ===="
# On utilise l'exécutable python du venv pour être 100% sûr
"./$VENV_DIR/bin/python" server_asr.py