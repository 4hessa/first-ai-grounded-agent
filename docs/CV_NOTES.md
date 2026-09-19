# CV and interview notes

## Suggested CV entry

**FirstAI Grounded Agent — Python, NVIDIA NIM, RAG, Agentic AI**

Built an Arabic-first AI-agent application around NVIDIA NIM with bounded tool calling, grounded retrieval, persistent user-controlled notes, multi-branch research, citation validation, resumable checkpoints, secret-conscious API handling, and 108 deterministic regression tests.

## Strong interview talking points

- Why source-ID validation prevents invented references but is not semantic fact checking.
- Why worker context isolation is different from operating-system sandboxing.
- How a global model-call budget prevents recovery logic from creating unbounded cost or loops.
- Why knowledge text is passed as untrusted evidence rather than instructions.
- How checkpoint/resume logic avoids silently rerunning already accepted work.
- Why API-key handling, redirect rejection, and remote-error sanitization belong in agent engineering.
- How Arabic lexical normalization was bounded to avoid pretending to be a full morphological analyzer.

## Accurate claims

You can claim that you designed, implemented, tested, and iteratively debugged the application and its reliability mechanisms.

Do not claim that the application itself provides OS-level sandboxing or that NVIDIA certified the project. If discussing the course relationship, describe NVIDIA DLI as the learning context and the repository as your independent portfolio implementation.
