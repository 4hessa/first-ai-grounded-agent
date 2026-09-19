# Live validation notes — 2026-09-19

These notes distinguish live provider observations from deterministic local tests.

## Environment observed during development

- Windows host
- Python 3.13.14
- Direct NVIDIA provider mode
- Core application version 0.1.5 plus language-repair hardening patches later incorporated into the 0.2.x portfolio tree
- Runtime-isolation tools reported absent by `doctor`: NemoClaw, OpenShell, Docker

## Grounded retrieval observation

A grounded Arabic question about context isolation versus runtime isolation retrieved four chunks and produced a cited answer with one model request. The application correctly stated that splitting work among agents does not automatically enable runtime isolation.

## Language-repair regression

Initial live responses placed a required Latin acronym in the same line as Arabic prose. The language checker detected this and a second model request was made. Early reviewer outputs were rejected with the local reason `mixed_language`.

The recovery path was hardened so that required Latin terms can be placed on a separate line while preserving protected content. A later live run produced a clean accepted form using `API` on its own line with exactly two model requests and no rejection warning.

## Clean-first-pass regression

A later prompt explicitly required the term `Agentic AI`. The first model response already placed the term on its own line, so the application accepted the answer with one model request and did not spend a language-repair request.

## Provider error observation

One separately submitted prompt received HTTP status 500 from the remote service. The transcript did not establish the root cause. The application surfaced the numeric status without exposing a provider response body or credential.

## Claim boundary

The 0.2.1 portfolio tree is validated offline by the deterministic 108-test suite and an installed-package smoke test. The live observations above validate the same core execution and language-repair paths from the development build, but they are not a formal benchmark of model quality, availability, runtime isolation, or the current provider service.
