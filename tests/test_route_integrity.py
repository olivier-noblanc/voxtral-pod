"""
Tests de régression structurels sur le routage FastAPI.
Détecte les routes fantômes, les stubs et les doubles mounts
avant qu'ils n'atteignent la prod.
"""
import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

os_environ_patch = {"TESTING": "1"}

import os
os.environ["TESTING"] = "1"
from backend.main import app

API_PY = Path(__file__).parents[1] / "backend" / "routes" / "api.py"
MAIN_PY = Path(__file__).parents[1] / "backend" / "main.py"


# ---------------------------------------------------------------------------
# 1. Pas de routes dupliquées (même method + path)
# ---------------------------------------------------------------------------
def test_no_duplicate_routes() -> None:
    """Deux routes avec le même method+path → FastAPI prend la première silencieusement."""
    seen: dict[str, str] = {}
    duplicates: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or []:
            key = f"{method} {route.path}"
            if key in seen:
                duplicates.append(f"{key} ('{seen[key]}' écrasé par '{route.name}')")
            else:
                seen[key] = route.name or "anonymous"

    assert not duplicates, (
        f"Routes dupliquées détectées — la première implémentation est silencieusement ignorée :\n"
        + "\n".join(duplicates)
    )


# ---------------------------------------------------------------------------
# 2. Pas de stubs retournant {"status": "ok"} sans logique
# ---------------------------------------------------------------------------
def test_no_stub_routes_in_api_py() -> None:
    """
    Détecte les fonctions stub dans api.py :
    fonctions dont le corps se résume à `return {"status": "ok"}` ou similaire.
    Ces stubs écrasent les vraies implémentations enregistrées par include_router().
    """
    source = API_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)

    stub_functions: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Corps de la fonction = une seule instruction Return
        body = [n for n in node.body if not isinstance(n, ast.Expr)]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        ret = body[0].value
        # Return d'un dict littéral avec uniquement "status"
        if isinstance(ret, ast.Dict):
            keys = [k.s if isinstance(k, ast.Constant) else None for k in ret.keys]
            if keys and all(k in ("status", "filename") for k in keys if k):
                stub_functions.append(node.name)

    assert not stub_functions, (
        f"Fonctions stub détectées dans api.py — supprimer ou implémenter :\n"
        + "\n".join(stub_functions)
    )


# ---------------------------------------------------------------------------
# 3. Un seul include_router dans main.py (pas de double mount)
# ---------------------------------------------------------------------------
def test_single_router_mount_in_main() -> None:
    """
    Un double app.include_router() crée des routes dupliquées.
    Le frontend n'utilise pas de prefix /api — un seul mount sans prefix suffit.
    """
    source = MAIN_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)

    include_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "include_router":
            # Récupérer le prefix si présent
            prefix = next(
                (kw.value.s for kw in node.keywords
                 if kw.arg == "prefix" and isinstance(kw.value, ast.Constant)),
                ""
            )
            prefix_str = str(prefix.decode() if isinstance(prefix, bytes) else prefix)
            include_calls.append(f"include_router(prefix='{prefix_str}')")

    assert len(include_calls) == 1, (
        f"Attendu 1 seul app.include_router(), trouvé {len(include_calls)} : {include_calls}\n"
        f"Le frontend appelle /route directement, jamais /api/route."
    )


# ---------------------------------------------------------------------------
# 4. Toutes les routes répondent (pas de 404/500 sur les routes GET sans params)
# ---------------------------------------------------------------------------
def test_static_get_routes_respond() -> None:
    """
    Les routes GET sans paramètres de path doivent répondre (pas 404, pas 500).
    Détecte les routes enregistrées mais non branchées sur une vraie implémentation.
    """
    client = TestClient(app, raise_server_exceptions=False)
    failures: list[str] = []

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if "GET" not in (route.methods or []):
            continue
        # Ignorer les routes avec paramètres de path
        if "{" in route.path:
            continue

        resp = client.get(route.path)
        if resp.status_code in (404, 500):
            failures.append(f"GET {route.path} → {resp.status_code}")

    assert not failures, (
        f"Routes GET sans params qui retournent 404/500 :\n" + "\n".join(failures)
    )
