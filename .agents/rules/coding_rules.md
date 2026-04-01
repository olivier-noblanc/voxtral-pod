---
trigger: always_on
---

# Cadrage LLM & Pipeline Multi-Langage

## Objectif
Garantir que le code généré par le LLM (Cline ou Antigravity) soit fiable, testé et conforme aux standards, même sur des projets SSR avec Python + HTML/CSS/JS.

## Règles générales
1. **Génération ciblée**
   - 1 prompt = 1 fonction / 1 classe / 1 module
   - Éviter de générer tout le projet en une seule requête
2. **Monolangage par génération**
   - Python (backend/SSR)
   - JS/CSS/HTML (frontend)
3. **TDD obligatoire**
   - Tests écrits avant le code
   - Tests unitaires, propriété (Hypothesis) ou snapshot
4. **Types et Linters**
   - Python → `mypy --strict`, `ruff`
   - JS → `ESLint`, `Jest` / `Vitest`
   - CSS → `Stylelint`
   - HTML → validateurs et snapshot tests
5. **Pas de side effects non documentés**
6. **Versionnement clair**
   - Branche/commit dédié par génération
   - Historisation pour rollback

## Workflow recommandé

### 1. Pipeline léger / pre-commit
- Pré-commit automatique à chaque `git commit` :
```bash id="pipeline-example"
ruff . --fix
mypy --strict .
pytest -q tests/unit
eslint src/js
stylelint src/css
html-validator-cli src/templates
```

---

## Règles de routage FastAPI — IMPÉRATIVES

Ces règles s'appliquent à chaque modification touchant les routes HTTP.

### Où déclarer les routes
- Les routes sont déclarées dans les **sub-routers métier** : `system.py`, `audio.py`, `transcriptions.py`, etc.
- `backend/routes/api.py` est un **agrégateur pur** : uniquement des `router.include_router(x.router)`.
- **Ne jamais ajouter** de `@router.get/post` dans `api.py`, sauf pour `/download_transcript` qui est une route composite documentée.

### Stubs et routes fantômes — INTERDITS
- Ne jamais créer de fonction `_dummy_*` ou stub `return {"status": "ok"}` pour faire passer un test.
- Si un test échoue faute de route → **corriger le test**, pas créer un stub.
- Un stub qui écrase une vraie implémentation est un bug silencieux.

### Mount dans main.py
- **Un seul** `app.include_router(api_module.router)` **sans prefix**.
- Ne jamais doubler avec un `prefix="/api"`.
- Le frontend (app.js) appelle `/change_model`, `/transcriptions`, etc. — jamais `/api/...`.
- Les tests doivent appeler les mêmes URLs que le frontend.

### Tests de routes
- Toujours utiliser les URLs sans prefix : `client.post("/change_model")` pas `client.post("/api/change_model")`.
- Le test `tests/test_routes_contract.py` scanne tous les sub-routers — le faire tourner avant tout commit touchant les routes.
