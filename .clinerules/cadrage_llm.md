# CADRAGE LLM : Quality Standards for voxtral-pod

To ensure maximum reliability and prevent the "LLM fake pass" effect, follow these strict rules for any code generation or test creation.

## 1. No Hardcoding for Tests
Tests must derive their assertions from logic or dynamic properties. Avoid writing tests that pass simply because you hardcoded a specific string in both the implementation and the test.

## 2. Mandatory Structural Validation (SSR)
Any new template or rendering logic must be accompanied by:
- A **Golden HTML test** (`pytest-snapshot`): Frizzing the visual/markup state.
- A **DOM Audit test** (`BeautifulSoup`): Verifying semantic elements (`main`, `nav`, etc.).

## 3. Boundary & Negative Testing
Every core function must include tests for:
- Invalid inputs (None, empty, malicious).
- Boundary conditions (very short/long audio, max timestamps).
- Property invariants (Hypothesis).

## 4. Mutation Testing Awareness
When writing tests for critical modules (`backend/utils.py`, `backend/core/`), run `mutmut` to ensure your tests actually kill the mutants. A passing test is not enough; it must fail if the logic changes.

## 5. Pipeline Consistency
Always run the full quality check before committing:
1. `ruff . --fix`
2. `mypy --strict .`
3. `pytest --cache-clear`

## 6. HTML/JS Separation
- No inline scripts or styles.
- Use `.js` and `.css` files.
- No CDNs (download assets locally).
