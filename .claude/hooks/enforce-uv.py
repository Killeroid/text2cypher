#!/usr/bin/env python3
"""Block bare pip/python/pytest/ruff commands; the project uses uv.

The command is tokenised with shlex, so quoted text (a commit message, an echo) is
never scanned, and each segment of a compound command (&&, ||, ;, |) is checked on
its own. A segment that starts with `uv` is always fine, e.g. `uv run python -m pytest`.
"""

from __future__ import annotations

import json
import shlex
import sys

SEPARATORS = {"&&", "||", ";", "|", "&"}


def segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    result: list[list[str]] = [[]]
    for token in lexer:
        if token in SEPARATORS:
            result.append([])
        else:
            result[-1].append(token)
    return [s for s in result if s]


def violation(tokens: list[str]) -> str | None:
    # Skip leading VAR=value assignments.
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if not tokens:
        return None
    first, rest = tokens[0], tokens[1:]
    if first in ("pip", "pip3"):
        return f"bare '{first}'. Use 'uv add' / 'uv remove'."
    if first in ("python", "python3"):
        if rest[:2] == ["-m", "pip"]:
            return f"'{first} -m pip'. Use 'uv add' / 'uv remove'."
        if rest[:2] == ["-m", "pytest"]:
            return f"'{first} -m pytest'. Use 'uv run pytest'."
        return f"bare '{first}'. Use 'uv run python'."
    if first in ("pytest", "ruff"):
        return f"bare '{first}'. Use 'uv run {first}'."
    return None


def main() -> int:
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    try:
        parsed = segments(command)
    except ValueError:  # unbalanced quotes etc.: let the shell report it
        return 0
    for tokens in parsed:
        if problem := violation(tokens):
            print(f"Blocked: {problem}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
