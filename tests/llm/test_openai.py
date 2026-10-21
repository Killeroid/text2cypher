import json

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from text2cypher.llm import Completion, Message, OpenAIClient, Usage

pytestmark = pytest.mark.anyio


pytestmark = pytest.mark.anyio


def _openai_client(
    reply: dict, requests: list[dict] | None = None
) -> tuple[AsyncOpenAI, list[dict]]:
    """A real ``AsyncOpenAI`` whose HTTP layer answers with ``reply``."""
    seen: list[dict] = [] if requests is None else requests

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )
    return client, seen


def _chat_reply(content: str | None, usage: dict | None) -> dict:
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        reply["usage"] = usage
    return reply


async def test_openai_client_sends_system_and_user_and_returns_text_and_usage():
    usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    client, requests = _openai_client(_chat_reply("```cypher\nRETURN 1\n```", usage))
    llm = OpenAIClient("test-model", client=client)

    completion = await llm.complete("be brief", "which movies?")

    assert completion == Completion("```cypher\nRETURN 1\n```", Usage(11, 5))
    assert requests == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "which movies?"},
            ],
            "temperature": 0,
        }
    ]


async def test_openai_client_omits_the_temperature_when_it_is_none():
    client, requests = _openai_client(_chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client, temperature=None)

    await llm.complete("s", "u")

    assert "temperature" not in requests[0]


async def test_openai_client_without_usage_or_content_gives_no_usage_and_empty_text():
    client, _ = _openai_client(_chat_reply(None, None))
    llm = OpenAIClient("test-model", client=client)

    assert await llm.complete("s", "u") == Completion("", None)


async def test_openai_client_can_send_one_user_message_for_a_model_without_a_system_role():
    client, requests = _openai_client(_chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client, system_role=False)

    await llm.complete("be brief", "which movies?")

    assert requests[0]["messages"] == [
        {"role": "user", "content": "be brief\n\nwhich movies?"}
    ]


async def test_openai_client_sends_every_turn_of_a_conversation_after_the_system_message():
    client, requests = _openai_client(_chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client)

    await llm.chat(
        "be brief",
        [
            Message("user", "which movies?"),
            Message("assistant", "```cypher\nRETURN 1\n```"),
            Message("user", "that has problems"),
        ],
    )

    assert requests[0]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "which movies?"},
        {"role": "assistant", "content": "```cypher\nRETURN 1\n```"},
        {"role": "user", "content": "that has problems"},
    ]


async def test_a_model_without_a_system_role_gets_the_system_text_in_the_first_turn():
    client, requests = _openai_client(_chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client, system_role=False)

    await llm.chat(
        "be brief",
        [
            Message("user", "which movies?"),
            Message("assistant", "RETURN 1"),
            Message("user", "that has problems"),
        ],
    )

    assert requests[0]["messages"] == [
        {"role": "user", "content": "be brief\n\nwhich movies?"},
        {"role": "assistant", "content": "RETURN 1"},
        {"role": "user", "content": "that has problems"},
    ]
