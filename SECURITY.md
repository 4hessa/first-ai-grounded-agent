# Security policy and threat boundaries

## Secrets

Never commit API keys. The application accepts `NVIDIA_API_KEY` from the local environment or a hidden terminal prompt. Runtime state and common secret files are ignored by Git.

If a key is exposed in a terminal transcript, screenshot, commit, issue, or chat, revoke it at the provider and create a new one.

## What the application does

- Allows only fixed API endpoint families.
- Rejects HTTP redirects in the direct client.
- Does not surface remote response bodies, headers, or reason phrases in application errors.
- Bounds request and response sizes.
- Treats retrieved knowledge as untrusted evidence.
- Allows only registered tools with exact argument schemas.
- Validates checkpoint/state paths and rejects unsafe file forms where supported.

## What the application does not do

- It does not provide operating-system sandboxing.
- It does not prove that an installed sandboxing tool is active or correctly configured.
- It does not guarantee model-output truthfulness.
- It does not provide a complete prompt-injection defense.
- It does not replace deployment-specific network, filesystem, identity, or secret-management policies.

## Reporting

For a public portfolio repository, avoid posting live credentials or sensitive environment details in an issue. Revoke exposed credentials first, then provide a minimal redacted reproduction.
