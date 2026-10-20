---
name: release-check
description: Run the full quality gate before a release. Use when the user wants to prepare a release, check if the project is ready to tag, or verify everything passes.
---

Run these checks in order and report the results:

1. `uv run ruff check .` (lint)
2. `uv run ruff format --check .` (format verification)
3. `uv run ty check` (type check)
4. `uv run lint-imports` (the layers of the package)
5. `uv run pytest -W error` (tests)
6. `uv run --group docs sphinx-build -W --keep-going -E -q docs /tmp/docs-check` (the documentation builds without warnings)
7. `env -u UV_PREVIEW uv lock --check` (the lock is the one CI's uv expects; never lock with `UV_PREVIEW` set)
8. Read the `version` field from `pyproject.toml` and confirm it matches the intended release version.

If any check fails, fix the issue and rerun that check before continuing. Report a summary: which checks passed, which failed, and the current version. Also check that `CHANGELOG.md` has a dated section for that version.