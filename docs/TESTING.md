# Testing strategy

The project uses Python's standard-library `unittest` framework so the core validation suite runs without third-party packages.

## Test groups

- `test_agent.py`: tool loop, tool-schema enforcement, bounded iterations, grounded-source validation, direct-search routing, API configuration and transport hygiene.
- `test_evidence.py`: Arabic retrieval regressions, evidence propagation, retrieved-vs-cited source behavior, and quality-note behavior.
- `test_quality.py`: plan validation, call-budget recovery, language-repair invariants, duplicate-term cleanup, and user-visible output regressions.
- `test_research.py`: worker/synthesis recovery, citation repair, safe diagnostics, model-call budget behavior.
- `test_resume.py`: checkpoint validation, concurrency claims, resume budgets, immutable prior artifacts, and service-status preservation.

## Commands

Full suite:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Portable quality gate:

```bash
python scripts/quality_gate.py
```

The quality gate compiles the package and runs the full deterministic test suite. Live provider tests remain manual because they require credentials, network access, provider availability, and can consume quota.
