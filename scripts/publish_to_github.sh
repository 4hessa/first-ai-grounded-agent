#!/usr/bin/env bash
set -euo pipefail

repo="${1:-4hessa/first-ai-grounded-agent}"

command -v gh >/dev/null 2>&1 || {
  echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2
  exit 1
}

gh auth status >/dev/null

if gh repo view "$repo" >/dev/null 2>&1; then
  echo "Repository already exists: $repo. Refusing to overwrite it." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init
  git add .
  git commit -m "Build portfolio-ready grounded Arabic AI agent"
fi

gh repo create "$repo" \
  --public \
  --description "Arabic-first grounded AI agent with NVIDIA NIM, bounded tools, RAG, research orchestration, citation validation, and 108 regression tests." \
  --source . \
  --remote origin \
  --push

echo "Published: https://github.com/$repo"
