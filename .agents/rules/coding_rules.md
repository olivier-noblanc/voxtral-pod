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