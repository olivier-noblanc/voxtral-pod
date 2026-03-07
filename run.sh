#!/bin/bash

# ==============================================================================
# Voxtral Pod Launcher v4.0 (The "Zero-Brain" Edition)
# Ce script gère tout : code, venv corrompu, dépendances et GPU.
# ==============================================================================

REPO_URL="https://github.com/olivier-noblanc/voxtral-pod.git"
VENV_DIR="venv_asr"
MODEL="${1:-voxtral}"

echo "===================================================="
echo "   🎙️  LANCEUR VOXTRAL-POD v4.0 (AUTO-REPAIR)  🎙️"
echo "===================================================="

# 1. Récupération & Réparation du code
git config --global --add safe.directory "$PWD"
if [ ! -d ".git" ]; then
    echo "[*] Initialisation du code..."
    git init .
    git remote add origin "$REPO_URL"
fi
git remote set-url origin "$REPO_URL" # On s'assure que l'URL est la bonne
echo "[*] Synchronisation GitHub..."
git fetch origin main && git reset --hard origin/main || echo "[!] Mode local (échec sync)"

# 2. Gestion de l'environnement virtuel (Auto-Correction)
# Si le venv existe mais n'a pas été créé avec --system-site-packages, on le recrée
# car sur Onyxia c'est indispensable pour Torch/CUDA.
FORCE_REINSTALL=false
if [ -d "$VENV_DIR" ]; then
    # Petit test : est-ce que le venv hérite bien des paquets système ?
    if ! "./$VENV_DIR/bin/python" -c "import torch" &> /dev/null; then
        echo "[!] Venv incomplet ou corrompu (Torch non trouvé). Nettoyage..."
        rm -rf "$VENV_DIR"
        FORCE_REINSTALL=true
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Création du venv avec accès aux paquets GPU système..."
    PYTHON_CMD="python3"
    command -v python3.11 &> /dev/null && PYTHON_CMD="python3.11"
    $PYTHON_CMD -m venv --system-site-packages "$VENV_DIR"
    FORCE_REINSTALL=true
fi

source "$VENV_DIR/bin/activate"

# 3. Installation des dépendances
if [ "$FORCE_REINSTALL" = true ] || ! pip show fastapi &> /dev/null; then
    echo "[*] Installation des dépendances (pip)..."
    pip install -U pip
    pip install -r requirements.txt || (echo "❌ Erreur critique lors de l'installation pip." && exit 1)
else
    echo "[*] Dépendances OK."
fi

# 4. Lancement
export ASR_MODEL="$MODEL"
echo "===================================================="
echo "🚀 LANCEMENT : Modèle=$MODEL"
echo "===================================================="
"./$VENV_DIR/bin/python" server_asr.py