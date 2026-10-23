"""The adapter that gives the library's chat protocol over the OpenAI SDK's Chat Completions API."""

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
from text2cypher.llm._options import DEFAULTS, ModelOptions, checked_options


class ChatCompletionsAdapter:
    """Asks a model through the SDK's Chat Completions API: a system message and the turns of the conversation.

    It is what an ``openai.AsyncOpenAI`` given to ``Text2Cypher`` or ``LLMSchemaFilter``
    (with ``model=``) becomes; it is not part of the public API.

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
        self._options: ModelOptions = {**DEFAULTS, **checked_options(model_options)}
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
    ) -> "ChatCompletionsAdapter":
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
