"""The real models of the format measurement, reached through an OpenAI-compatible endpoint.

The settings are in ``.env`` at the root of the repository (git-ignored), one ``NAME: value`` or
``NAME=value`` to a line:

- ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``, which the OpenAI SDK reads by itself
- ``CHEAP_MODEL`` and ``EXPENSIVE_MODEL``, the names the endpoint knows them by
- ``EXPENSIVE_MODEL_REASONING_EFFORT``, sent with every call to the expensive model

Nothing here prints a setting, and an error names a missing setting, never a value. A variable
that is already in the environment wins over the file.
"""

import os
import re
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI

from text2cypher.llm._chat_completions import ChatCompletionsAdapter

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
REQUIRED = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CHEAP_MODEL",
    "EXPENSIVE_MODEL",
    "EXPENSIVE_MODEL_REASONING_EFFORT",
)
type Which = Literal["cheap", "expensive"]

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*?)\s*$")


def parse_env(text: str) -> dict[str, str]:
    """The settings in ``text``: blank lines and ``#`` comments are skipped, and one pair
    of quotes around a value is removed."""
    settings = {}
    for line in text.splitlines():
        found = None if line.lstrip().startswith("#") else _LINE.match(line)
        if found:
            value = found[2]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            settings[found[1]] = value
    return settings


def load_env(path: Path = ENV_FILE) -> None:
    """Put the settings of ``path`` in the environment, unless they are already there.

    Raises:
        RuntimeError: Naming the settings that are missing, if there are any.
    """
    if path.exists():
        for name, value in parse_env(path.read_text(encoding="utf-8")).items():
            os.environ.setdefault(name, value)
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"missing settings (in {path.name} or the environment): {', '.join(missing)}"
        )


def model_name(which: Which) -> str:
    return os.environ["CHEAP_MODEL" if which == "cheap" else "EXPENSIVE_MODEL"]


def make_client(which: Which) -> ChatCompletionsAdapter:
    """The library's own client for the cheap or the expensive model.

    The expensive one is asked with its reasoning effort, and with no temperature, which
    reasoning models do not accept. The cheap one is asked with temperature 0, as the library
    does by default.
    """
    load_env()
    sdk = AsyncOpenAI()  # the key and the endpoint come from the environment
    if which == "cheap":
        return ChatCompletionsAdapter(
            model_name(which),
            client=sdk,
            model_options={"temperature": 0, "max_retries": 5},
        )
    return ChatCompletionsAdapter(
        model_name(which),
        client=sdk,
        model_options={
            "temperature": None,
            "max_retries": 5,
            "reasoning_effort": os.environ["EXPENSIVE_MODEL_REASONING_EFFORT"],
        },
    )
