"""What the rest of the library needs from an LLM."""

from dataclasses import dataclass
from typing import NamedTuple, Protocol


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


class AsyncLLMClient(Protocol):
    """A stateless chat model: every call is one system and one user message."""

    async def complete(self, system: str, user: str) -> Completion:
        """Ask the model, with a system message and a user message, and return its reply."""
        ...
