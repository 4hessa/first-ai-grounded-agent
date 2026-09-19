param(
    [string]$Repository = "4hessa/first-ai-grounded-agent"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Install it from https://cli.github.com/ and run gh auth login first."
}

& gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI is not authenticated. Run: gh auth login"
}

& gh repo view $Repository *> $null
if ($LASTEXITCODE -eq 0) {
    throw "Repository already exists: $Repository. This script refuses to overwrite an existing repository."
}

if (-not (Test-Path ".git")) {
    & git init
    & git add .
    & git commit -m "Build portfolio-ready grounded Arabic AI agent"
}

& gh repo create $Repository `
    --public `
    --description "Arabic-first grounded AI agent with NVIDIA NIM, bounded tools, RAG, research orchestration, citation validation, and 108 regression tests." `
    --source . `
    --remote origin `
    --push

Write-Host "Published: https://github.com/$Repository"
