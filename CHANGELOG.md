# Changelog

## 0.2.1 - 2026-09-19

Packaging and CI hardening release.

- Added an installable `first-ai` console command.
- Bundled the safe default configuration and sample knowledge corpus for installed-package use.
- Moved installed runtime state to a per-user directory instead of relying on writes inside `site-packages`.
- Added the `FIRST_AI_STATE_DIR` environment override for installed runtime state.
- Added package-data synchronization checks to the offline quality gate.
- Added installed-package smoke testing to CI on Python 3.12 and 3.13.
- Added a repository line-ending policy and ignored local build artifacts.

## 0.2.0 - 2026-09-19

Portfolio hardening release.

- Added conservative Arabic/Latin language-repair recovery and regression coverage.
- Added cleanup for reviewer wrapper text and duplicated required acronyms.
- Preserved citation, URL, code, and numeric invariants after language repair.
- Added public-safe sample knowledge corpus and validated configuration.
- Added English and Arabic documentation, architecture notes, security boundaries, CI, and portable quality gate.
- Verified 108 deterministic tests locally.

## 0.1.5 - 2026-09-19

- Added bounded language review, stricter research-plan validation, and progress messages.
- Improved citation and recovery behavior across grounded and research workflows.
