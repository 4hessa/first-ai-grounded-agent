# Architecture

## Design goals

FirstAI is intentionally small enough to audit. The design prioritizes bounded behavior, visible provenance, deterministic validation, and explicit failure over feature breadth.

## Components

### CLI

`first_ai.__main__` exposes offline diagnostics, live chat, grounded Q&A, branched research, semantic indexing, resume, and local-note management. The CLI prints host-generated progress messages rather than provider payloads.

### Agent

`first_ai.agent.Agent` owns routing, model-call budgets, tool dispatch, grounded generation, research planning, worker execution, synthesis, citation repair, and language repair. Model outputs are considered untrusted until they pass local checks relevant to the workflow.

### Retrieval

`first_ai.knowledge.Corpus` loads bounded `.md` and `.txt` files, chunks them, derives stable source IDs, supports lexical retrieval, and optionally validates and consumes a dense index.

Lexical Arabic handling is deliberately bounded. It normalizes Unicode variants and expands a small set of attached articles; it is not a morphological analyzer.

### API client

`first_ai.api.NvidiaClient` uses fixed HTTPS destinations for direct NVIDIA access and a fixed local inference route for the OpenShell mode. It bounds request/response sizes, validates configuration, rejects redirects, and sanitizes transport errors.

### Storage

`first_ai.storage.Store` persists user-approved notes, semantic indexes, branch drafts, failure reports, and resume claims. State is local and excluded from version control.

## Workflows

### General agent loop

1. Validate user input and bounded history.
2. Send messages plus the allowlisted tool schema to the model.
3. If the model requests a tool, validate tool name and exact argument schema.
4. Execute the local tool and return its result as tool data.
5. Repeat within the configured request budget.
6. Validate and display the final answer.

### Grounded answer

1. Search local knowledge before generation.
2. If no evidence matches, return without asking the model.
3. Send retrieved evidence as untrusted data.
4. Require at least one valid retrieved source reference in the answer.
5. Reject unknown or fabricated reference formats.
6. Optionally run one language repair if mixed Arabic/Latin prose is detected and budget remains.
7. Revalidate protected invariants after repair.

### Branched research

1. Generate a strict JSON plan containing two or three tasks.
2. Validate the plan locally; one bounded repair may be used if budget permits.
3. Retrieve evidence per task.
4. Execute workers with fresh contexts; workers do not receive sibling histories.
5. Persist accepted worker drafts and full retrieved evidence.
6. Synthesize from the validated union of branch evidence.
7. Allow one shared correction path under the global budget.
8. Persist a failure checkpoint when safe resume is possible.
9. Resume synthesis without silently rerunning workers, retrieval, or embeddings.

## Trust boundaries

### Trusted application logic

- Tool allowlist and argument validation
- Request budgets and size limits
- Source-ID generation and matching
- Checkpoint schema and state-file validation
- Local invariant checks

### Untrusted inputs

- User text
- Model output
- Knowledge-file text
- Remote service errors and payloads
- Saved state that fails schema/integrity checks

## Context isolation vs runtime isolation

Research workers have isolated *model contexts*. This prevents ordinary sibling-history sharing inside the orchestration design. It does not isolate operating-system processes, files, environment variables, network destinations, or credentials.

Runtime isolation belongs to the deployment layer and must be configured and tested separately. The `doctor` command reports tool presence only; it does not claim that a sandbox is active.

## Known limitations

- The retrieval validator verifies citation provenance, not semantic entailment.
- Lexical Arabic normalization is intentionally limited.
- Semantic retrieval depends on external embedding service availability and quality.
- The CLI stores no chat transcript by default, but live prompts and required context are sent to the selected model provider.
- Language repair is a formatting-quality mechanism, not a semantic equivalence proof.
- OpenShell/NemoClaw deployment is outside the default local runtime.
