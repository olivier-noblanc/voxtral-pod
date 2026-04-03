"""
Teste qu'aucune balise <script> n'est présente dans les fichiers Python du projet.
Cette règle garantit que le JavaScript est placé dans des fichiers *.js afin de pouvoir
être linté séparément, conformément aux directives du projet.
"""

import pathlib
import re
from typing import Generator

import pytest


def _python_files() -> Generator[pathlib.Path, None, None]:
    """
    Retourne un générateur de chemins vers tous les fichiers *.py du projet, 
    en excluant les tests et les chemins ignorés par .gitignore.
    """
    root = pathlib.Path(__file__).resolve().parents[1]  # dossier racine du projet
    # Charger les patterns .gitignore
    gitignore_path = root / ".gitignore"
    ignore_patterns = set()
    if gitignore_path.is_file():
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.add(line.rstrip("/"))
    def is_ignored(p: pathlib.Path) -> bool:
        # Vérifier chaque composant du chemin contre les patterns simples
        for part in p.relative_to(root).parts:
            if part in ignore_patterns:
                return True
        return False
    # Exclure le répertoire tests et les chemins ignorés
    return (p for p in root.rglob("*.py") if "tests" not in p.parts and not is_ignored(p))


SCRIPT_TAG_PATTERN = re.compile(r"<script>|<script>", re.IGNORECASE)


@pytest.mark.parametrize("py_file", list(_python_files()))
def test_no_script_tags_in_python(py_file: pathlib.Path) -> None:
    """Vérifie qu'aucune balise <script> n'est présente dans le fichier Python donné."""
    content = py_file.read_text(encoding="utf-8")
    matches = SCRIPT_TAG_PATTERN.search(content)
    assert matches is None, f"Balise <script> trouvée dans {py_file}"
