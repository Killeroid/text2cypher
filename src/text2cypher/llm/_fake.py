"""A scripted LLM for tests."""

from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple

from text2cypher.llm._base import Completion, Message, Tool


class Call(NamedTuple):
    """What an LLM was asked once.

    Attributes:
        system: The system message.
        user: The last user message: the one that asks.
        messages: Every turn that was sent; ``complete`` sends one.
        tools: The tools that were offered, if any.
        tool_choice: ``"auto"``, or ``"none"`` when a call was forbidden.
    """

    system: str
    user: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...] = ()
    tool_choice: Literal["auto", "none"] = "auto"


class FakeLLM:
    """Answers with the given replies in order, and records every call.

    A reply is a ``Completion`` (give it ``tool_calls`` to script a request for a tool), or a
    plain string for one without usage, or an exception, which is raised at that call (to
    test what happens when the model fails). It has both ``chat`` and ``complete``; the
    library calls ``chat``. ``FakeLLM(replies, tools=False)`` says it cannot be offered tools,
    as a client without that support does.

    Attributes:
        calls: What it was asked, in order.
    """

    def __init__(
        self, replies: Iterable[str | Completion | Exception], *, tools: bool = True
    ) -> None:
        self._replies = iter(replies)
        self.calls: list[Call] = []
        self.supports_tools = tools

    async def complete(self, system: str, user: str) -> Completion:
        """Ask with one user message; see ``chat``."""
        return await self.chat(system, [Message("user", user)])

    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> Completion:
        """Record the call and return the next reply.

        Raises:
            RuntimeError: If every reply has been used.
        """
        self.calls.append(
            Call(
                system,
                messages[-1].content,
                tuple(messages),
                tuple(tools or ()),
                tool_choice,
            )
        )
        try:
            reply = next(self._replies)
        except StopIteration:
            raise RuntimeError(
                f"FakeLLM has no reply left for call {len(self.calls)}"
            ) from None
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, Completion) else Completion(reply, None)
