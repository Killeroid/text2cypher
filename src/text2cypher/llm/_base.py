"""What the rest of the library needs from an LLM."""

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol


@dataclass(frozen=True, slots=True)
class Usage:
    """The tokens one LLM call used; ``+`` adds two.

    Attributes:
        input_tokens: Tokens in the messages sent.
        output_tokens: Tokens in the reply.
    """

    input_tokens: int
    output_tokens: int

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class Tool:
    """A function the model may ask to have called.

    It has the shape the OpenAI SDK, MCP and the other tool-calling APIs share: a name, a
    description, and the parameters as a JSON Schema object.

    Attributes:
        name: The name the model calls it by.
        description: What it does and when to call it, for the model.
        parameters: The arguments as a JSON Schema object, for example
            ``{"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}``.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model's request to call a tool.

    Attributes:
        id: The call's id; the answer to it carries the same id.
        name: The name of the tool.
        arguments: The arguments, as a dict (an OpenAI-style reply has them as a JSON string,
            which the client parses; an invalid string gives an empty dict).
    """

    id: str
    name: str
    arguments: Mapping[str, Any]


class Completion(NamedTuple):
    """One reply: its text, the tokens it used when the provider says so, and tool calls.

    Attributes:
        text: The reply, with the query in a fenced block (an empty block declines).
        usage: The tokens the call used, or ``None`` if they are not known.
        tool_calls: The tools the model asked to have called, if it was offered any. The
            text is then usually empty.
        finish_reason: Why the reply ended, if the client knows: ``"stop"`` (it finished),
            ``"length"`` (it hit a token limit), ``"content_filter"`` (the provider blocked
            it), ``"refusal"`` (the model refused to answer), ``"tool_calls"``, or the
            provider's own word. ``None`` is not known. It tells an empty reply that is a
            refusal from one that is a limit or a filter.
    """

    text: str
    usage: Usage | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


class Message(NamedTuple):
    """One turn of a conversation.

    Attributes:
        role: ``"user"`` for what the library asks, ``"assistant"`` for what the model
            replied earlier, ``"tool"`` for the answer to one of its tool calls.
        content: The text of the turn.
        tool_calls: For an assistant turn, the tool calls the model made in it.
        tool_call_id: For a ``"tool"`` turn, the id of the call it answers.
    """

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


class AsyncChatClient(Protocol):
    """A chat model that is given the conversation so far, and may be offered tools.

    This is the method the library calls when a client has it: a correction is then
    the earlier queries the model wrote, each followed by what was wrong with it. A
    client that has only ``complete`` still works (see ``AsyncLLMClient``).

    ``tools`` are functions the model may ask to have called; the reply then has
    ``tool_calls``, and the library answers them with ``"tool"`` messages and asks again.
    ``tool_choice="none"`` offers them but forbids a call, to force a text answer. A client
    that cannot use tools may ignore both, and says so with ``supports_tools = False``.
    """

    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> Completion:
        """Ask the model, with a system message and the turns so far (the first is from
        the user), and return its reply."""
        ...


def supports_tools(llm: object) -> bool:
    """Whether ``llm`` can be offered tools.

    A client says so with a ``supports_tools`` attribute. Without one, it can if its
    ``chat`` takes a ``tools`` argument, or any keyword arguments.
    """
    declared = getattr(llm, "supports_tools", None)
    if isinstance(declared, bool):
        return declared
    chat = getattr(llm, "chat", None)
    if not callable(chat):
        return False
    try:
        parameters = inspect.signature(chat).parameters
    except (TypeError, ValueError):
        return False
    return "tools" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class AsyncLLMClient(Protocol):
    """A stateless chat model: every call is one system and one user message.

    A client with this method alone is given a conversation as one user message that
    lists the earlier attempts; add ``chat`` (see ``AsyncChatClient``) to be given real
    turns.
    """

    async def complete(self, system: str, user: str) -> Completion:
        """Ask the model, with a system message and a user message, and return its reply."""
        ...
