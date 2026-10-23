"""An LLM client for OpenAI's Chat Completions API (and compatible servers)."""

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, cast

from openai import NOT_GIVEN, AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from text2cypher.llm._base import Completion, Message, Tool, ToolCall, Usage

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
    """The settings of how a model is asked, for ``OpenAIClient(model_options=...)``.

    Every key is optional. Asking a new client, a key left out takes its default; for
    ``OpenAIClient.with_model``, it keeps the value the client has.

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


_DEFAULTS: ModelOptions = {
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
    unknown = (options or {}).keys() - _DEFAULTS.keys()
    if unknown:
        raise ValueError(
            f"model_options has no {', '.join(sorted(unknown))}: "
            f"the settings are {', '.join(_DEFAULTS)}"
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


class OpenAIClient:
    """Asks an OpenAI chat model with a system message and the turns of the conversation.

    Args:
        model: The model to ask. There is no default: the right one depends on the
            account and on the schema being prompted.
        client: The ``AsyncOpenAI`` to use, e.g. one with another ``base_url`` for a
            compatible server. By default one is made from the environment.
        model_options: How to ask it: ``temperature``, ``reasoning_effort``, ``timeout``
            and the others of ``ModelOptions``. Left out, the defaults are used.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI | None = None,
        model_options: ModelOptions | None = None,
    ) -> None:
        self._model = model
        self._sdk = client or AsyncOpenAI()
        self._options: ModelOptions = {**_DEFAULTS, **checked_options(model_options)}
        self._apply()

    def _apply(self) -> None:
        options = self._options
        timeout, retries = options["timeout"], options["max_retries"]
        # The copy shares the connection pool of the client it is made from
        self._client = self._sdk.with_options(
            timeout=NOT_GIVEN if timeout is None else timeout,
            max_retries=NOT_GIVEN if retries is None else retries,
        )
        self.supports_tools = options["tools"]

    def with_model(
        self, model: str, model_options: ModelOptions | None = None
    ) -> "OpenAIClient":
        """A client that asks ``model`` instead, with the same SDK client and every setting
        but the ones in ``model_options``, for example a cheaper model for a part of the work.

        A model that thinks differently may need other settings: a key you leave out is
        kept, and ``None`` takes a setting away (``{"reasoning_effort": None}``).

        Raises:
            ValueError: If ``model_options`` has an unknown key, or ``extra`` sets a field the
                client sets itself.
        """
        other = copy.copy(self)
        other._model = model
        other._options = {**self._options, **checked_options(model_options)}
        other._apply()
        return other

    async def complete(self, system: str, user: str) -> Completion:
        """Ask the model with one user message; see ``chat``."""
        return await self.chat(system, [Message("user", user)])

    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> Completion:
        """Ask the model, and return its reply, the tool calls it made, and when the API
        reports it, the tokens used.

        ``tools`` are offered as functions. A call whose arguments are not valid JSON has
        empty arguments, so that the answer can say what is missing and the model can try
        again; calls of any kind other than a function are ignored.
        """
        offered: Any = [_tool_param(tool) for tool in tools or ()] or omit
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(system, messages),
            temperature=_omit_none(self._options["temperature"]),
            tools=offered,
            tool_choice=tool_choice if tools else omit,
            reasoning_effort=_omit_none(self._options["reasoning_effort"]),
            extra_body=dict(self._options["extra"] or {}) or None,
        )
        used = response.usage
        message = response.choices[0].message
        return Completion(
            text=message.content or "",
            usage=None
            if used is None
            else Usage(used.prompt_tokens, used.completion_tokens),
            tool_calls=_tool_calls(message),
            finish_reason=_finish_reason(response.choices[0].finish_reason, message),
        )

    def _messages(
        self, system: str, turns: Sequence[Message]
    ) -> list[ChatCompletionMessageParam]:
        if not self._options["system_role"]:
            first = Message("user", f"{system}\n\n{turns[0].content}")
            return [_param(first), *map(_param, turns[1:])]
        return [{"role": "system", "content": system}, *map(_param, turns)]


def _omit_none(value: Any) -> Any:
    return omit if value is None else value


def _param(turn: Message) -> ChatCompletionMessageParam:
    if turn.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": turn.tool_call_id or "",
            "content": turn.content,
        }
    if turn.role == "assistant" and turn.tool_calls:
        return {
            "role": "assistant",
            "content": turn.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in turn.tool_calls
            ],
        }
    if turn.role == "user":
        return {"role": "user", "content": turn.content}
    return {"role": "assistant", "content": turn.content}


def _tool_param(tool: Tool) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        },
    }


def _tool_calls(message: ChatCompletionMessage) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(call.id, call.function.name, _arguments(call.function.arguments))
        for call in message.tool_calls or ()
        if call.type == "function"
    )


def _finish_reason(reason: str | None, message: ChatCompletionMessage) -> str | None:
    """The API's finish reason, with a refusal (a message with no text and a ``refusal``) named as one."""
    if message.refusal and not message.content:
        return "refusal"
    return reason


def _arguments(text: str) -> dict[str, Any]:
    try:
        arguments = json.loads(text)
    except ValueError:
        return {}
    return arguments if isinstance(arguments, dict) else {}
