"""
test_warnings_fixes.py
----------------------
Tests de non-régression pour les 5 avertissements 🟡 corrigés :
  1. f-string no_gpu correctement interpolée dans la home route
  2. init_db() appelé uniquement via startup event (pas au module-level)
  3. _assemble_chunks() fonctionne correctement (I/O sync extraite)
  4. audio_np recalculé une seule fois dans LiveSession.save_audio_file
  5. ProxyHeadersMiddleware a des trusted_hosts configurés
"""
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Pré-charger l'app pour éviter un circular import au moment du patch
from backend.main import app


# ── 1. f-string no_gpu dans la home route ───────────────────────────────────

def test_home_ws_url_no_gpu_true():
    """Quand no_gpu=True, l'URL WS doit contenir partial_albert=true."""
    fake_engine = MagicMock()
    fake_engine.no_gpu = True

    with patch("backend.routes.api.get_asr_engine", return_value=fake_engine), \
         patch("backend.routes.api.get_current_model", return_value="whisper"):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "partial_albert=true" in resp.text, (
            "ws_url doit contenir partial_albert=true quand no_gpu=True"
        )


def test_home_ws_url_no_gpu_false():
    """Quand no_gpu=False, l'URL WS doit contenir partial_albert=false."""
    fake_engine = MagicMock()
    fake_engine.no_gpu = False

    with patch("backend.routes.api.get_asr_engine", return_value=fake_engine), \
         patch("backend.routes.api.get_current_model", return_value="whisper"):
        from backend.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "partial_albert=false" in resp.text, (
            "ws_url doit contenir partial_albert=false quand no_gpu=False"
        )


def test_home_ws_url_not_literal_no_gpu():
    """L'URL WS ne doit PAS contenir la chaîne littérale '{no_gpu}'."""
    with patch("backend.routes.api.HTML_UI", "TEMPLATE {{ws_url}}"):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/")
        assert "{no_gpu}" not in resp.text, "f-string jamais interpolée détectée !"


# ── 2. state.py : init_db() plus au module-level ────────────────────────────

def test_state_module_no_init_db_at_import():
    """
    Vérifier que init_db() n'est plus appelé au niveau module dans state.py.
    On inspecte le source AST plutôt que d'exécuter — plus fiable.
    """
    import ast, pathlib

    src = pathlib.Path("backend/state.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Chercher les Call statements au niveau module (hors classes/fonctions)
    top_level_calls = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    call_names = []
    for node in top_level_calls:
        call = node.value
        if isinstance(call.func, ast.Name):
            call_names.append(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            call_names.append(call.func.attr)

    assert "init_db" not in call_names, (
        f"init_db() est encore appelé au module-level ! Calls trouvés: {call_names}"
    )


# ── 3. _assemble_chunks() — test unitaire I/O ───────────────────────────────

def test_assemble_chunks(tmp_path):
    """_assemble_chunks doit concaténer et supprimer les chunks."""
    from backend.routes.api import _assemble_chunks

    # Créer des chunks factices
    for i in range(3):
        chunk = tmp_path / f"chunk_{i:04d}"
        chunk.write_bytes(f"data{i}".encode())

    assembled = str(tmp_path / "audio_full.wav")
    _assemble_chunks(assembled, 3, str(tmp_path))

    # Le fichier assemblé doit exister et contenir les 3 parties
    result = open(assembled, "rb").read()
    assert result == b"data0data1data2"

    # Les chunks doivent avoir été supprimés
    for i in range(3):
        assert not (tmp_path / f"chunk_{i:04d}").exists()


# ── 4. audio_np non doublé dans LiveSession ─────────────────────────────────

def test_save_audio_file_no_duplicate_audio_np():
    """
    Vérifier via AST qu'audio_np n'est pas assigné deux fois dans save_audio_file.
    """
    import ast, pathlib

    src = pathlib.Path("backend/core/live.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "save_audio_file":
            assignments = [
                n for n in ast.walk(node)
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "audio_np"
                    for t in n.targets
                )
            ]
            count = len(assignments)
            assert count == 1, (
                f"audio_np est assigné {count} fois dans save_audio_file, attendu 1"
            )
            return

    pytest.fail("Fonction save_audio_file introuvable dans live.py")


# ── 5. ProxyHeadersMiddleware avec trusted_hosts ────────────────────────────

def test_proxy_middleware_has_trusted_hosts():
    """
    Vérifier via AST que ProxyHeadersMiddleware est appelé avec trusted_hosts.
    """
    import ast, pathlib

    src = pathlib.Path("backend/main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_middleware"
        ):
            keywords = {kw.arg for kw in node.keywords}
            # Chercher un appel add_middleware avec ProxyHeadersMiddleware
            args_names = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    args_names.append(arg.id)
            if "ProxyHeadersMiddleware" in args_names:
                assert "trusted_hosts" in keywords, (
                    "ProxyHeadersMiddleware doit avoir trusted_hosts explicite"
                )
                return

    pytest.fail("add_middleware(ProxyHeadersMiddleware, ...) non trouvé dans main.py")
