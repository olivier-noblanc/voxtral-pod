# Reste à faire (TODO) - Voxtral-Pod

## 1. Type Annotations & Linting (MyPy/Ruff)
- [ ] **Tests Annotations** : Il reste environ 400+ tests à typer strictement (`-> None`, fixtures, mocks) pour passer `mypy --strict tests/`.
- [ ] **Ruff Cleanup Repository-wide** : S'assurer que chaque module respecte parfaitement les règles ruff (notamment les imports inutilisés ou mal placés après la modularisation).

## 2. Optimisation & Performance
- [ ] **Batch Processing Optimisation** : Tester la stabilité de la diarisation sur des fichiers très longs (> 1 heure) sur CPU.
- [ ] **VAD Calibration** : Affiner les seuils de Silero VAD pour réduire le bruit dans les environnements peu clairs.

## 3. Design & AESTHETICS (UI/UX)
- [ ] **Refonte du Design SSR** : Moderniser les templates Jinja2 (`index.html`, `view.html`, `postprocess.html`) avec un thème premium (dark mode par défaut, TailwindCSS, micro-animations).
- [ ] **Visualiseur de Diarisation** : Améliorer le `diarization_view.html` pour inclure des graphes d'activité (Timeline) plutôt qu'un simple tableau.

## 4. Fonctionnalités Albert (IA)
- [ ] **Multi-modèles Albert** : Permettre le choix explicite entre différents LLM pour le compte rendu (Albert Large, Small, etc.).
- [ ] **Export Multi-formats** : Ajouter l'export PDF et DOCX pour les versions nettoyées.

## 5. Déploiement & CI/CD
- [ ] **GitHub Actions** : Mettre en place un pipeline de CI qui fait tourner `mypy --strict` et `pytest` sur chaque PR.
