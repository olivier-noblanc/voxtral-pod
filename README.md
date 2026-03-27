# Voxtral‑Pod – Vérification du contrat DOM

Ce dépôt utilise **pre‑commit** pour s’assurer que le JavaScript (`static/app.js`) et le HTML généré (`backend/html_ui.py`) restent synchronisés.

## Installation du hook pre‑commit

```bash
# Installer le framework pre‑commit (une fois)
pip install pre-commit

# Installer les hooks définis dans le dépôt
pre-commit install
```

## Prérequis obligatoires

Le projet dépend des bibliothèques listées dans **requirements.txt**.  
Avant de lancer l’application, installez‑les :

```bash
pip install -r requirements.txt
```

> **⚠️ Aucun fallback n’est autorisé** – si une dépendance est manquante, le démarrage doit échouer afin que vous puissiez l’installer correctement.

Le hook `check-dom-contract` sera exécuté automatiquement à chaque `git commit` :

* Il analyse les IDs référencés dans le JavaScript (`getElementById`, `querySelector('#…')`).
* Il compare ces IDs avec ceux présents dans le HTML (`id="…"`, `id='…'`).
* Il signale les IDs manquants ainsi que les anti‑patterns :
  * `addEventListener` placé dans un `if (el)` sans journalisation.
  * `ws.send` appelé sans vérification `ws.readyState === 1`.

Le commit échoue (code de sortie 1) si des problèmes sont détectés.

## Exécuter les tests

```bash
pytest -q
```

Le test `tests/test_dom_contract.py` reproduit la même logique que le script et garantit que le contrat DOM reste valide.

## Dépannage

* Si le hook signale un ID manquant, ajoutez‑le soit dans le HTML (`backend/html_ui.py`), soit retirez‑le du JavaScript.
* Pour désactiver temporairement le hook : `git commit --no-verify`.

---

*Ce fichier a été généré automatiquement par le script `scripts/check_dom_contract.py`.*