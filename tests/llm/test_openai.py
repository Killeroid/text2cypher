import pytest
from openai import InternalServerError
from support import chat_reply, openai_client

from text2cypher.llm import Completion, Message, OpenAIClient, Usage

pytestmark = pytest.mark.anyio


pytestmark = pytest.mark.anyio


async def test_openai_client_sends_system_and_user_and_returns_text_and_usage():
    usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    client, requests = openai_client(chat_reply("```cypher\nRETURN 1\n```", usage))
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
    client, requests = openai_client(chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client, temperature=None)

    await llm.complete("s", "u")

    assert "temperature" not in requests[0]


async def test_openai_client_without_usage_or_content_gives_no_usage_and_empty_text():
    client, _ = openai_client(chat_reply(None, None))
    llm = OpenAIClient("test-model", client=client)

    assert await llm.complete("s", "u") == Completion("", None)


async def test_openai_client_can_send_one_user_message_for_a_model_without_a_system_role():
    client, requests = openai_client(chat_reply("x", None))
    llm = OpenAIClient("test-model", client=client, system_role=False)

    await llm.complete("be brief", "which movies?")

    assert requests[0]["messages"] == [
        {"role": "user", "content": "be brief\n\nwhich movies?"}
    ]


async def test_openai_client_sends_every_turn_of_a_conversation_after_the_system_message():
    client, requests = openai_client(chat_reply("x", None))
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
    client, requests = openai_client(chat_reply("x", None))
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
    client, _ = openai_client(chat_reply("x", None), timeouts=timeouts)

    await OpenAIClient("test-model", client=client).complete("s", "u")

    assert timeouts == [120.0]


async def test_the_timeout_can_be_set_and_none_leaves_the_clients_own():
    timeouts: list[float | None] = []
    client, _ = openai_client(chat_reply("x", None), timeouts=timeouts, timeout=7)

    await OpenAIClient("test-model", client=client, timeout=30).complete("s", "u")
    await OpenAIClient("test-model", client=client, timeout=None).complete("s", "u")

    assert timeouts == [30.0, 7.0]


async def test_a_failing_call_is_retried_as_many_times_as_asked():
    client, requests = openai_client(chat_reply("ok", None), failures=1)

    completion = await OpenAIClient(
        "test-model", client=client, max_retries=1
    ).complete("s", "u")

    assert completion.text == "ok"
    assert len(requests) == 2


async def test_with_no_retries_the_first_failure_is_the_error():
    client, requests = openai_client(chat_reply("ok", None), failures=1)

    with pytest.raises(InternalServerError):
        await OpenAIClient("test-model", client=client, max_retries=0).complete(
            "s", "u"
        )

    assert len(requests) == 1


async def test_the_clients_own_retries_stay_unless_the_option_is_given():
    # two failures, then an answer: the SDK's two retries are enough
    client, requests = openai_client(chat_reply("ok", None), failures=2)

    completion = await OpenAIClient("test-model", client=client).complete("s", "u")

    assert completion.text == "ok"
    assert len(requests) == 3
