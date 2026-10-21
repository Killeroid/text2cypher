import json
from typing import Any

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI, InternalServerError

from text2cypher.llm import Completion, Message, OpenAIClient, Usage

pytestmark = pytest.mark.anyio


pytestmark = pytest.mark.anyio


def _openai_client(
    reply: dict,
    requests: list[dict] | None = None,
    *,
    failures: int = 0,
    timeouts: list[float | None] | None = None,
    **options: Any,
) -> tuple[AsyncOpenAI, list[dict]]:
    """A real ``AsyncOpenAI`` whose HTTP layer answers with ``reply``, after ``failures`` 500s.

    The read timeout each request was sent with is added to ``timeouts``.
    """
    seen: list[dict] = [] if requests is None else requests
    failed = 0

    def answer(request: httpx.Request) -> httpx.Response:
        nonlocal failed
        seen.append(json.loads(request.content))
        if timeouts is not None:
            timeouts.append(request.extensions["timeout"]["read"])
        if failed < failures:
            failed += 1
            return httpx.Response(500, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
        **options,
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


async def test_a_call_gives_up_after_two_minutes_by_default_and_not_after_ten():
    timeouts: list[float | None] = []
    client, _ = _openai_client(_chat_reply("x", None), timeouts=timeouts)

    await OpenAIClient("test-model", client=client).complete("s", "u")

    assert timeouts == [120.0]


async def test_the_timeout_can_be_set_and_none_leaves_the_clients_own():
    timeouts: list[float | None] = []
    client, _ = _openai_client(_chat_reply("x", None), timeouts=timeouts, timeout=7)

    await OpenAIClient("test-model", client=client, timeout=30).complete("s", "u")
    await OpenAIClient("test-model", client=client, timeout=None).complete("s", "u")

    assert timeouts == [30.0, 7.0]


async def test_a_failing_call_is_retried_as_many_times_as_asked():
    client, requests = _openai_client(_chat_reply("ok", None), failures=1)

    completion = await OpenAIClient(
        "test-model", client=client, max_retries=1
    ).complete("s", "u")

    assert completion.text == "ok"
    assert len(requests) == 2


async def test_with_no_retries_the_first_failure_is_the_error():
    client, requests = _openai_client(_chat_reply("ok", None), failures=1)

    with pytest.raises(InternalServerError):
        await OpenAIClient("test-model", client=client, max_retries=0).complete(
            "s", "u"
        )

    assert len(requests) == 1


async def test_the_clients_own_retries_stay_unless_the_option_is_given():
    # two failures, then an answer: the SDK's two retries are enough
    client, requests = _openai_client(_chat_reply("ok", None), failures=2)

    completion = await OpenAIClient("test-model", client=client).complete("s", "u")

    assert completion.text == "ok"
    assert len(requests) == 3
