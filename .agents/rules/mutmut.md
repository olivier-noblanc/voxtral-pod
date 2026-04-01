---
trigger: always_on
---

You must follow these rules strictly:

- Do not hardcode values to satisfy tests
- All outputs must come from real logic
- HTML must be valid and parsable (BeautifulSoup compliant)
- Code must handle invalid inputs explicitly
- Code must pass property-based tests (Hypothesis)
- Code must survive mutation testing (mutmut)

If a test passes incorrectly, consider it a failure.
Do not optimize for test passing, optimize for correctness.

Mypy expects that all references to names have a corresponding definition in an active scope, such as an assignment, function definition or an import. This can catch missing definitions, missing imports, and typos.

mypy requires that all functions have annotations (either a Python 3 annotation or a type comment).