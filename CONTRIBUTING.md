# Contributing

1. Keep changes small and scoped.
2. Add or update a regression test for behavior changes.
3. Do not add secrets, private knowledge files, provider response bodies, or runtime state.
4. Run `python scripts/quality_gate.py` before submitting a change.
5. Document any change that alters a trust boundary, request budget, source-validation rule, or persistence format.
6. Preserve the distinction between model-context isolation and runtime sandboxing.

Commit messages should describe the behavior changed rather than only the file edited.
