"""How a model is asked: the settings of ``model_options``, their defaults and their checks."""

from collections.abc import Mapping
from typing import Any, TypedDict, cast

# What the library sets on every request itself, so ``extra`` may not
_RESERVED = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "tools",
        "tool_choice",
        "reasoning_effort",
        "stream",
    }
)


class ModelOptions(TypedDict, total=False):
    """The settings of how a model is asked, for the ``model_options`` of ``Text2Cypher`` and
    ``LLMSchemaFilter``.

    Every key is optional: a key left out takes its default.

    Attributes:
        temperature: The sampling temperature; ``None`` leaves it out, for models that only
            accept their own (reasoning models). Default ``0``.
        system_role: ``False`` puts the system text before the first user message's text,
            for a model that has no system role. Default ``True``.
        timeout: Seconds to wait for a reply before giving up on the call (it is then
            retried, see ``max_retries``). The OpenAI SDK waits 10 minutes by default, and a
            question can make a dozen calls, so a hung connection would hold a run for hours;
            the default is 2 minutes. Raise it for a model that thinks for longer. It applies
            to an ``AsyncOpenAI`` you pass too; ``None`` leaves that client's own timeout.
        max_retries: How many times the SDK retries a call that fails on a timeout, a
            connection error, a rate limit or a server error, with a pause between. ``None``
            (the default) leaves the client's own setting (2 for one made here). When it
            still fails, the run ends with an ``LLMError``.
        tools: ``False`` says the model or server cannot be offered tools (some compatible
            servers cannot), so the library does without them. Default ``True``.
        reasoning_effort: How much a reasoning model thinks (``"minimal"``, ``"low"``,
            ``"medium"``, ``"high"``, or what the server takes). ``None`` (the default)
            leaves it out.
        extra: Other request fields to send on every call, such as ``max_completion_tokens``,
            ``seed`` or ``service_tier``; a name the SDK does not know is sent as it is.
            ``model``, ``messages``, ``temperature``, ``tools``, ``tool_choice``,
            ``reasoning_effort`` and ``stream`` are set by the client and are an error here.
            A new ``extra`` replaces the old one as a whole.
    """

    temperature: float | None
    system_role: bool
    timeout: float | None
    max_retries: int | None
    tools: bool
    reasoning_effort: str | None
    extra: Mapping[str, Any] | None


DEFAULTS: ModelOptions = {
    "temperature": 0,
    "system_role": True,
    "timeout": 120.0,
    "max_retries": None,
    "tools": True,
    "reasoning_effort": None,
    "extra": None,
}


def checked_options(options: Mapping[str, Any] | None) -> ModelOptions:
    """``options`` as they are, after checking the names and what ``extra`` sets.

    Raises:
        ValueError: If a key is not a setting of ``ModelOptions``, or ``extra`` sets a field
            the client sets itself.
    """
    unknown = (options or {}).keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(
            f"model_options has no {', '.join(sorted(unknown))}: "
            f"the settings are {', '.join(DEFAULTS)}"
        )
    taken = _RESERVED.intersection((options or {}).get("extra") or ())
    if taken:
        raise ValueError(
            f"extra cannot set {', '.join(sorted(taken))}: the client sets them"
        )
    checked = dict(options or {})
    if checked.get("extra") is not None:
        checked["extra"] = dict(
            checked["extra"]
        )  # not the caller's own, who may change it
    return cast("ModelOptions", checked)
