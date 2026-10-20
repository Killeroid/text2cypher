#!/usr/bin/env bash
# Auto-fix and format an edited .py file. Lint errors ruff cannot fix are sent to
# stderr with exit 2, so Claude sees them instead of them being silently dropped.
input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path')

if [[ "$file_path" == *.py ]] && [[ -f "$file_path" ]]; then
  remaining=$(uv run ruff check --fix --quiet "$file_path" 2>&1)
  status=$?
  uv run ruff format --quiet "$file_path" >/dev/null 2>&1
  if [[ $status -ne 0 ]]; then
    echo "$remaining" >&2
    exit 2
  fi
fi
exit 0
