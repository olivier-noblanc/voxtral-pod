"""
Teste qu'aucune balise <script> ne charge un fichier depuis un CDN.
Le projet doit héberger toutes les ressources JavaScript en local afin de pouvoir
les linté et de ne pas dépendre d’Internet.
"""

import pathlib
import re

import pytest


def _all_files():
    """Retourne un générateur de chemins vers tous les fichiers du projet (sauf les dossiers virtuels)."""
    root = pathlib.Path(__file__).resolve().parents[1]  # dossier racine du projet
    # Exclure les dossiers typiquement ignorés (venv, __pycache__, .git, etc.)
    ignore_dirs = {".git", "__pycache__", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
    for path in root.rglob("*"):
        if path.is_file() and not any(part in ignore_dirs for part in path.parts):
            yield path


# Recherche d'un attribut src pointant vers un URL http(s) dans une balise <script>
CDN_SCRIPT_PATTERN = re.compile(r'<script\s+[^>]*src=["\']https?://[^"\']+["\']', re.IGNORECASE)


@pytest.mark.parametrize("file_path", _all_files())
def test_no_cdn_script_src(file_path: pathlib.Path):
    """Vérifie qu'aucune balise <script src=\"https://...\"> n'est présente dans le fichier donné."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        # Ignorer les fichiers non textuels ou non lisibles
        return

    match = CDN_SCRIPT_PATTERN.search(content)
    assert match is None, f"CDN script trouvé dans {file_path}: {match.group(0)}"
