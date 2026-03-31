import pathlib


def _files_with_extension(root: pathlib.Path, extensions):
    for path in root.rglob("*"):
        if path.suffix.lower() in extensions and path.is_file():
            yield path

def _load_gitignore_patterns(root: pathlib.Path):
    """
    Charge les motifs d'ignorés depuis le fichier .gitignore à la racine du projet.
    Les lignes vides et les commentaires sont ignorés.
    """
    gitignore_path = root / ".gitignore"
    patterns = []
    if gitignore_path.is_file():
        for line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns

def test_no_vibe_coding_markers() -> None:
    """
    Vérifie qu'aucun fichier source (.py, .js, .css) ne contient la chaîne
    « TRACE DE RECHERCHE », qui est une trace de l'outil de remplacement de code
    (vibe coding) et ne doit pas rester dans le code final.
    Les fichiers listés dans .gitignore sont exclus de la vérification.
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
    offending_files = []
    for p in root.rglob("*.py"):
        if "tests" in p.parts or is_ignored(p):
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "-" * 7 + " SEARCH" in content:
            offending_files.append(str(p.relative_to(root)))
    assert not offending_files, (
        "Des fichiers contiennent des traces de vibe coding : "
        f"{', '.join(offending_files)}"
    )
