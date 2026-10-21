"""What the rest of the library needs from an LLM."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol


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


class Completion(NamedTuple):
    """One reply: its text, and the tokens it used when the provider says so.

    Attributes:
        text: The reply, with the query in a fenced block (an empty block declines).
        usage: The tokens the call used, or ``None`` if they are not known.
    """

    text: str
    usage: Usage | None


class Message(NamedTuple):
    """One turn of a conversation.

    Attributes:
        role: ``"user"`` for what the library asks, ``"assistant"`` for what the model
            replied earlier.
        content: The text of the turn.
    """

    role: Literal["user", "assistant"]
    content: str


class AsyncChatClient(Protocol):
    """A chat model that is given the conversation so far.

    This is the method the library calls when a client has it: a correction is then
    the earlier queries the model wrote, each followed by what was wrong with it. A
    client that has only ``complete`` still works (see ``AsyncLLMClient``).
    """

    async def chat(self, system: str, messages: Sequence[Message]) -> Completion:
        """Ask the model, with a system message and the turns so far (the first and the
        last are from the user), and return its reply."""
        ...


class AsyncLLMClient(Protocol):
    """A stateless chat model: every call is one system and one user message.

    A client with this method alone is given a conversation as one user message that
    lists the earlier attempts; add ``chat`` (see ``AsyncChatClient``) to be given real
    turns.
    """

    async def complete(self, system: str, user: str) -> Completion:
        """Ask the model, with a system message and a user message, and return its reply."""
        ...
