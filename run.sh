#!/bin/bash

# ==============================================================================
# Voxtral Pod Launcher v3.0 (Super-Bootstrap)
# Ce script est conçu pour être copié-collé seul dans un dossier vide sur le Pod.
# Il se charge de tout : code, environnement, dépendances et lancement.
# ==============================================================================

REPO_URL="https://github.com/olivier-noblanc/voxtral-pod.git"
VENV_DIR="venv_asr"
PORT=8000
MODEL="${1:-voxtral}"

echo "===================================================="
echo "   🎙️  LANCEUR VOXTRAL-POD AUTOMATIQUE  🎙️"
echo "===================================================="

# 1. Récupération du code (Auto-réparation)
# Fix pour les environnements comme Onyxia/VSCode qui râlent sur les permissions
git config --global --add safe.directory "$PWD"

if [ ! -d ".git" ]; then
    echo "[*] Initialisation du dépôt Git..."
    git init .
fi

# On s'assure que le remote 'origin' est bien configuré (au cas où l'étape précédente a été interrompue)
git remote remove origin 2>/dev/null
git remote add origin "$REPO_URL"

echo "[*] Synchronisation avec GitHub (Force)..."
git fetch origin main
git checkout -f main
git reset --hard origin/main || (echo "❌ Erreur: Synchronisation impossible. Vérifie l'accès au repo." && exit 1)

# 2. Vérification de Python 3.11+
PYTHON_CMD="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
fi

# 3. Setup de l'environnement virtuel
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Création de l'environnement virtuel ($PYTHON_CMD) avec accès aux paquets système..."
    # --system-site-packages permet d'utiliser torch/cuda déjà installés sur le pod
    $PYTHON_CMD -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 4. Installation/Vérification des dépendances
# On vérifie la présence de torch pour gagner du temps au reboot
if ! python -c "import torch" &> /dev/null; then
    echo "[!] Dépendances manquantes. Installation (cela peut prendre quelques minutes)..."
    pip install -U pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "❌ requirements.txt introuvable !"
        exit 1
    fi
else
    echo "[*] Environnement Python OK."
fi

# 5. Lancement du serveur
export ASR_MODEL="$MODEL"
echo "===================================================="
echo "🚀 LANCEMENT : Modèle=$MODEL | Port=$PORT"
echo "===================================================="

# On s'assure d'utiliser le python du venv
"./$VENV_DIR/bin/python" server_asr.py