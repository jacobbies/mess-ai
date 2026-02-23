# Public Release Checklist

This checklist focuses on two release gates:
- secret/data hygiene
- outsider-facing documentation clarity

## 1) Secret And Data Audit

Run these checks from repo root before making a public push:

```bash
# 1. Confirm sensitive/local files are not tracked.
git ls-files | rg '^(data/|mlruns/|mlflow.db|\.env|.*\.pem$|.*\.key$|.*id_rsa)'

# 2. Scan tracked files for common credential patterns.
git ls-files -z | xargs -0 rg -n --no-heading \
  --glob '!.github/workflows/*.yml' \
  --glob '!docs/PUBLIC_RELEASE_CHECKLIST.md' \
  "(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----|aws_access_key_id|aws_secret_access_key|(?i)api[_-]?key\s*[:=]|(?i)secret[_-]?key\s*[:=]|(?i)password\s*[:=]|(?i)token\s*[:=])"

# 3. Ensure generated probing results are not staged accidentally.
git status --short
```

Expected:
- no credential values found
- no generated local artifacts staged (`mlruns/`, `mlflow.db`, `mess/probing/layer_discovery_results.json`)

Current repository notes:
- `data/metadata/music_catalog.csv` and `data/metadata/music_catalog.json` are tracked as metadata tables.
- large dataset binaries/features remain excluded by `.gitignore`.

## 2) Documentation For External Users

Before public release, verify:
- README has install instructions for local dev and external dependency use.
- README clearly separates maintained scripts from experimental scripts.
- README includes minimal Python API examples for search-only consumers.
- `scripts/_NEEDS_UPDATE.txt` is kept accurate when script status changes.

## 3) Recommended Final Pre-Public Commands

```bash
uv run pytest -q
uv run ruff check .
uv run mypy mess
```

If lint/type debt is intentionally deferred, state that explicitly in release notes.
