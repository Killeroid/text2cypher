"""A scripted LLM for tests."""

from collections.abc import Iterable
from typing import NamedTuple

from text2cypher.llm._base import Completion


class Call(NamedTuple):
    """What an LLM was asked once.

    Attributes:
        system: The system message.
        user: The user message.
    """

    system: str
    user: str


class FakeLLM:
    """Answers with the given replies in order, and records every call.

    A reply is a ``Completion``, or a plain string for one without usage.

    Attributes:
        calls: What it was asked, in order.
    """

    def __init__(self, replies: Iterable[str | Completion]) -> None:
        self._replies = iter(replies)
        self.calls: list[Call] = []

    async def complete(self, system: str, user: str) -> Completion:
        """Record the call and return the next reply.

        Raises:
            RuntimeError: If every reply has been used.
        """
        self.calls.append(Call(system, user))
        try:
            reply = next(self._replies)
        except StopIteration:
            raise RuntimeError(
                f"FakeLLM has no reply left for call {len(self.calls)}"
            ) from None
        return reply if isinstance(reply, Completion) else Completion(reply, None)
