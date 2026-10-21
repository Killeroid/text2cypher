"""An LLM client for OpenAI's Chat Completions API (and compatible servers)."""

import copy
import json
from collections.abc import Sequence
from typing import Any, Literal

from openai import NOT_GIVEN, AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from text2cypher.llm._base import Completion, Message, Tool, ToolCall, Usage


class OpenAIClient:
    """Asks an OpenAI chat model with a system message and the turns of the conversation.

    A model that has no system role (some served by compatible servers) needs
    ``system_role=False``: the system text then goes first in the first user message.

    Args:
        model: The model to ask. There is no default: the right one depends on the
            account and on the schema being prompted.
        client: The ``AsyncOpenAI`` to use, e.g. one with another ``base_url`` for a
            compatible server. By default one is made from the environment.
        temperature: ``None`` leaves it out, for models that only accept their own.
        system_role: ``False`` puts the system text before the first user message's text,
            for a model that has no system role.
        timeout: Seconds to wait for a reply before giving up on the call (it is then
            retried, see ``max_retries``). The OpenAI SDK waits 10 minutes by default, and a
            question can make a dozen calls, so a hung connection would hold a run for hours;
            this is 2 minutes. Raise it for a model that thinks for longer. It applies to a
            ``client`` you pass too; ``None`` leaves that client's own timeout.
        max_retries: How many times the SDK retries a call that fails on a timeout, a
            connection error, a rate limit or a server error, with a pause between. ``None``
            leaves the client's own setting (2 for one made here). When it still fails, the
            run ends with an ``LLMError``.
        tools: ``False`` says the model or server cannot be offered tools (some compatible
            servers cannot), so the library does without them.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI | None = None,
        temperature: float | None = 0,
        system_role: bool = True,
        timeout: float | None = 120.0,
        max_retries: int | None = None,
        tools: bool = True,
    ) -> None:
        self.supports_tools = tools
        self._model = model
        # The copy shares the connection pool of the client it is made from
        self._client = (client or AsyncOpenAI()).with_options(
            timeout=NOT_GIVEN if timeout is None else timeout,
            max_retries=NOT_GIVEN if max_retries is None else max_retries,
        )
        self._temperature = omit if temperature is None else temperature
        self._system_role = system_role

    def with_model(self, model: str) -> "OpenAIClient":
        """A client that asks ``model`` instead, with every other setting and the same SDK
        client, for example a cheaper model for a part of the work."""
        other = copy.copy(self)
        other._model = model
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
            temperature=self._temperature,
            tools=offered,
            tool_choice=tool_choice if tools else omit,
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
        if not self._system_role:
            first = Message("user", f"{system}\n\n{turns[0].content}")
            return [_param(first), *map(_param, turns[1:])]
        return [{"role": "system", "content": system}, *map(_param, turns)]


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
