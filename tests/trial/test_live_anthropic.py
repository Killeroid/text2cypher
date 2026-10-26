"""The adapter against the real Anthropic API: skipped unless a key and a model are in the environment.

Set ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_MODEL`` (for example ``claude-sonnet-5-5``), and run
``uv run pytest tests/trial/test_live_anthropic.py``. It asks for a few short replies, so it costs
very little. It exists because the adapter's tests use a mock transport, and so show that the
requests are shaped as the documentation says, not that the API accepts them. It is not run in CI.
"""

import os

import pytest
from anthropic import AsyncAnthropic

from text2cypher.llm import Message, Tool
from text2cypher.llm._messages import MessagesAdapter

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_MODEL")),
        reason="needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL",
    ),
]

LOOKUP = Tool(
    "lookup",
    "Look up what a code stands for. Always call it before answering.",
    {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
)


def adapter() -> MessagesAdapter:
    return MessagesAdapter(os.environ["ANTHROPIC_MODEL"], client=AsyncAnthropic())


async def test_a_question_is_answered_and_the_tokens_are_counted():
    completion = await adapter().complete(
        "Answer in one word.", "What colour is grass?"
    )

    assert completion.text.strip()
    assert completion.usage is not None
    assert completion.finish_reason == "stop"


async def test_a_text_reply_sent_back_without_its_thinking_is_accepted():
    llm = adapter()
    first = await llm.complete("Answer in one word.", "What colour is grass?")

    second = await llm.chat(
        "Answer in one word.",
        [
            Message("user", "What colour is grass?"),
            Message("assistant", first.text),
            Message("user", "And the sky?"),
        ],
    )

    assert second.text.strip()


async def test_a_tool_call_is_answered_and_the_thinking_goes_back_with_it():
    llm = adapter()
    system = "Call the lookup tool for the code in the question, then say what it stands for."
    turns = [Message("user", "What does code 7 stand for?")]

    first = await llm.chat(system, turns, [LOOKUP])
    assert first.tool_calls, "the model was expected to call the tool"
    turns += [
        Message("assistant", first.text, first.tool_calls),
        *(
            Message("tool", "7 stands for seven", tool_call_id=call.id)
            for call in first.tool_calls
        ),
    ]

    second = await llm.chat(system, turns, [LOOKUP])

    assert "seven" in second.text.lower()
