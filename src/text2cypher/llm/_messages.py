"""The adapter that gives the library's chat protocol over the Anthropic SDK's Messages API."""

import copy
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal

from anthropic import NOT_GIVEN, AsyncAnthropic, omit
from anthropic.types import Message as AnthropicMessage

from text2cypher.llm._base import Completion, Message, Tool, ToolCall, Usage
from text2cypher.llm._options import DEFAULTS, ModelOptions, checked_options

# The model's thinking counts toward it, so it is not small: the API requires a value
_MAX_TOKENS = 16000
# What the API takes from the client, so ``extra`` may not set it (``max_tokens`` it may)
_RESERVED = frozenset({"system", "top_p", "top_k"})
# How many replies that asked for tools are remembered (see ``_remember``)
_REMEMBERED = 256

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "refusal",
}


class MessagesAdapter:
    """Asks a model through the SDK's Messages API: a system prompt and the turns of the conversation.

    It is what an ``anthropic.AsyncAnthropic`` given to ``Text2Cypher`` or ``LLMSchemaFilter``
    (with ``model=``) becomes; it is not part of the public API.

    The models take no ``temperature``, ``top_p`` or ``top_k`` (the API refuses any value, or
    any but the default, depending on its version), so none is ever sent and a ``temperature``
    in ``model_options`` is an error: a reply cannot be made repeatable.

    Thinking is on by default and cannot be turned off on every model. A reply that asks for
    tools has thinking blocks, which the API requires back unchanged with the tool results, so
    the adapter keeps the content blocks of such a reply and sends them back. A reply that
    asks for nothing is sent back as its text alone, which the API allows.

    Args:
        model: The model to ask. There is no default: the right one depends on the
            account and on the schema being prompted.
        client: The ``AsyncAnthropic`` to use. By default one is made from the environment.
        model_options: How to ask it: ``reasoning_effort`` (sent as ``output_config.effort``),
            ``timeout``, ``max_retries``, ``tools`` and ``extra`` of ``ModelOptions``;
            ``extra`` may set ``max_tokens`` (default 16000). ``system_role`` has no effect: the
            system prompt is always its own field.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncAnthropic | None = None,
        model_options: ModelOptions | None = None,
    ) -> None:
        self._model = model
        self._sdk = client or AsyncAnthropic()
        self._options: ModelOptions = _checked(
            {**DEFAULTS, "temperature": None, **checked_options(model_options)}
        )
        # The blocks of replies that asked for tools, by the id of their first tool call
        self._remembered: OrderedDict[str, list[Any]] = OrderedDict()
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
    ) -> "MessagesAdapter":
        """A client that asks ``model`` instead, with the same SDK client and every setting
        but the ones in ``model_options``, for example a cheaper model for a part of the work.

        A model that thinks differently may need other settings: a key you leave out is
        kept, and ``None`` takes a setting away (``{"reasoning_effort": None}``).

        Raises:
            ValueError: If ``model_options`` has an unknown key, a ``temperature``, or an
                ``extra`` that sets a field the client sets itself.
        """
        other = copy.copy(self)
        other._model = model
        other._options = _checked({**self._options, **checked_options(model_options)})
        # Thinking blocks belong to the model that wrote them
        other._remembered = OrderedDict()
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
        """Ask the model, and return its reply, the tool calls it made, and the tokens used.

        ``tools`` are offered with their parameters as the input schema. The answers to a
        reply's tool calls are sent as the results of one user message.
        """
        extra = dict(self._options["extra"] or {})
        max_tokens = extra.pop("max_tokens", _MAX_TOKENS)
        effort = self._options["reasoning_effort"]
        offered: Any = [_tool_param(tool) for tool in tools or ()] or omit
        choice: Any = {"type": tool_choice} if tools else omit
        output: Any = {"effort": effort} if effort is not None else omit
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=self._messages(messages),
            tools=offered,
            tool_choice=choice,
            output_config=output,
            extra_body=extra or None,
        )
        return self._completion(response)

    def _completion(self, response: AnthropicMessage) -> Completion:
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = tuple(
            ToolCall(block.id, block.name, _input(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        if calls:
            self._remember(calls[0].id, list(response.content))
        used = response.usage
        reason = response.stop_reason
        return Completion(
            text=text,
            usage=Usage(
                used.input_tokens
                + (used.cache_creation_input_tokens or 0)
                + (used.cache_read_input_tokens or 0),
                used.output_tokens,
            ),
            tool_calls=calls,
            finish_reason=None
            if reason is None
            else _FINISH_REASONS.get(reason, reason),
        )

    def _remember(self, call_id: str, blocks: list[Any]) -> None:
        """Keep a reply's blocks, thinking included, for when its tool calls come back.

        A run of the library asks a handful of times, so the oldest are forgotten.
        """
        self._remembered[call_id] = blocks
        while len(self._remembered) > _REMEMBERED:
            self._remembered.popitem(last=False)

    def _messages(self, turns: Sequence[Message]) -> list[Any]:
        params: list[Any] = []
        for turn in turns:
            if turn.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": turn.tool_call_id or "",
                    "content": turn.content,
                }
                # Every answer to one reply's calls goes in the one user message
                if (
                    params
                    and params[-1]["role"] == "user"
                    and _only_results(params[-1])
                ):
                    params[-1]["content"].append(block)
                else:
                    params.append({"role": "user", "content": [block]})
            elif turn.role == "assistant" and turn.tool_calls:
                kept = self._remembered.get(turn.tool_calls[0].id)
                params.append(
                    {
                        "role": "assistant",
                        "content": kept
                        if kept is not None
                        else _blocks(turn.content, turn.tool_calls),
                    }
                )
            else:
                params.append({"role": turn.role, "content": turn.content})
        return params


def _checked(options: ModelOptions) -> ModelOptions:
    if options.get("temperature") is not None:
        raise ValueError(
            "temperature cannot be set for an anthropic.AsyncAnthropic: the models take no "
            "temperature (the API refuses it), so a reply cannot be made repeatable"
        )
    taken = _RESERVED.intersection(options.get("extra") or ())
    if taken:
        raise ValueError(
            f"extra cannot set {', '.join(sorted(taken))}: the client sets it, "
            "or the models take none"
        )
    return options


def _only_results(params: dict[str, Any]) -> bool:
    content = params["content"]
    return isinstance(content, list) and all(
        block["type"] == "tool_result" for block in content
    )


def _blocks(text: str, calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    """The blocks of a reply that asked for tools, when what it said is not remembered."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(call.arguments),
        }
        for call in calls
    )
    return blocks


def _tool_param(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


def _input(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
