import anthropic
import pytest
from support import anthropic_client, message_reply, tool_use_reply

from text2cypher.llm import Completion, Message, Tool, ToolCall, Usage, _messages
from text2cypher.llm._messages import MessagesAdapter

pytestmark = pytest.mark.anyio

QUERY = "```cypher\nRETURN 1\n```"
LOOKUP = Tool(
    "lookup",
    "Find a thing",
    {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
)


async def test_the_adapter_sends_the_system_prompt_and_the_user_message_and_returns_text_and_usage():
    sdk, requests = anthropic_client(
        message_reply(QUERY, {"input_tokens": 11, "output_tokens": 5})
    )
    llm = MessagesAdapter("test-model", client=sdk)

    completion = await llm.complete("be brief", "which movies?")

    assert completion == Completion(QUERY, Usage(11, 5), (), "stop")
    assert requests == [
        {
            "model": "test-model",
            "max_tokens": 16000,
            "system": "be brief",
            "messages": [{"role": "user", "content": "which movies?"}],
        }
    ]


async def test_no_sampling_setting_is_ever_sent():
    sdk, requests = anthropic_client(message_reply("x"))

    await MessagesAdapter("test-model", client=sdk).complete("s", "u")

    assert {"temperature", "top_p", "top_k"}.isdisjoint(requests[0])


async def test_the_tokens_read_from_and_written_to_the_cache_count_as_input():
    usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 7,
    }
    sdk, _ = anthropic_client(message_reply("x", usage))

    completion = await MessagesAdapter("m", client=sdk).complete("s", "u")

    assert completion.usage == Usage(20, 4)


async def test_the_text_is_the_text_blocks_and_not_the_thinking():
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "```cypher\n"},
        {"type": "text", "text": "RETURN 1\n```"},
    ]
    sdk, _ = anthropic_client(message_reply(blocks))

    completion = await MessagesAdapter("m", client=sdk).complete("s", "u")

    assert completion.text == QUERY


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "refusal"),
        ("pause_turn", "pause_turn"),
    ],
)
async def test_why_a_reply_ended_is_said_in_the_words_of_the_library(
    stop_reason: str, finish_reason: str
):
    sdk, _ = anthropic_client(message_reply([], None, stop_reason))

    completion = await MessagesAdapter("m", client=sdk).complete("s", "u")

    assert completion.text == ""
    assert completion.finish_reason == finish_reason


async def test_the_turns_of_a_conversation_are_sent_in_order():
    sdk, requests = anthropic_client(message_reply("x"))
    turns = [
        Message("user", "q"),
        Message("assistant", "first"),
        Message("user", "that was wrong"),
    ]

    await MessagesAdapter("m", client=sdk).chat("s", turns)

    assert requests[0]["messages"] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "that was wrong"},
    ]


async def test_tools_are_offered_with_their_parameters_as_the_input_schema():
    sdk, requests = anthropic_client(message_reply("x"))

    await MessagesAdapter("m", client=sdk).chat(
        "s", [Message("user", "q")], [LOOKUP], tool_choice="none"
    )

    assert requests[0]["tools"] == [
        {
            "name": "lookup",
            "description": "Find a thing",
            "input_schema": LOOKUP.parameters,
        }
    ]
    assert requests[0]["tool_choice"] == {"type": "none"}


async def test_without_tools_there_is_no_tool_choice_either():
    sdk, requests = anthropic_client(message_reply("x"))

    await MessagesAdapter("m", client=sdk).chat("s", [Message("user", "q")])

    assert "tools" not in requests[0]
    assert "tool_choice" not in requests[0]


async def test_a_reply_that_asks_for_tools_has_them_as_calls_with_their_arguments():
    sdk, _ = anthropic_client(
        tool_use_reply([("toolu_1", "lookup", {"id": "a"}), ("toolu_2", "lookup", {})])
    )

    completion = await MessagesAdapter("m", client=sdk).chat(
        "s", [Message("user", "q")], [LOOKUP]
    )

    assert completion.tool_calls == (
        ToolCall("toolu_1", "lookup", {"id": "a"}),
        ToolCall("toolu_2", "lookup", {}),
    )
    assert completion.finish_reason == "tool_calls"


async def test_the_answers_to_tool_calls_go_in_one_user_message_with_the_thinking_back_unchanged():
    sdk, requests = anthropic_client(
        tool_use_reply(
            [("toolu_1", "lookup", {"id": "a"}), ("toolu_2", "lookup", {"id": "b"})],
            thinking="signed",
        ),
        replies=None,
    )
    llm = MessagesAdapter("m", client=sdk)
    first = await llm.chat("s", [Message("user", "q")], [LOOKUP])
    turns = [
        Message("user", "q"),
        Message("assistant", first.text, first.tool_calls),
        Message("tool", "result a", tool_call_id="toolu_1"),
        Message("tool", "result b", tool_call_id="toolu_2"),
    ]

    await llm.chat("s", turns, [LOOKUP])

    assistant, results = requests[1]["messages"][1:]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == [
        "thinking",
        "tool_use",
        "tool_use",
    ]
    assert assistant["content"][0]["signature"] == "signed"
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result a"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "result b"},
        ],
    }


async def test_a_reply_that_asked_for_tools_and_is_not_remembered_is_rebuilt_from_its_calls():
    sdk, requests = anthropic_client(message_reply("x"))
    turns = [
        Message("user", "q"),
        Message("assistant", "reading", (ToolCall("toolu_9", "lookup", {"id": "a"}),)),
        Message("tool", "done", tool_call_id="toolu_9"),
    ]

    await MessagesAdapter("m", client=sdk).chat("s", turns, [LOOKUP])

    assert requests[0]["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "reading"},
            {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "lookup",
                "input": {"id": "a"},
            },
        ],
    }


async def test_a_reply_that_asked_for_nothing_is_sent_back_as_its_text_alone():
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": QUERY},
    ]
    sdk, requests = anthropic_client(message_reply(blocks))
    llm = MessagesAdapter("m", client=sdk)
    first = await llm.chat("s", [Message("user", "q")])

    await llm.chat(
        "s",
        [
            Message("user", "q"),
            Message("assistant", first.text),
            Message("user", "again"),
        ],
    )

    assert requests[1]["messages"][1] == {"role": "assistant", "content": QUERY}


async def test_only_the_latest_replies_that_asked_for_tools_are_remembered(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_messages, "_REMEMBERED", 1)
    sdk, requests = anthropic_client(
        message_reply("x"),
        replies=[
            tool_use_reply([("toolu_1", "lookup", {})], thinking="one"),
            tool_use_reply([("toolu_2", "lookup", {})], thinking="two"),
        ],
    )
    llm = MessagesAdapter("m", client=sdk)
    for _ in range(2):
        await llm.chat("s", [Message("user", "q")], [LOOKUP])

    def replay(call: str) -> list[str]:
        turns = [
            Message("user", "q"),
            Message("assistant", "", (ToolCall(call, "lookup", {}),)),
            Message("tool", "r", tool_call_id=call),
        ]
        content = llm._messages(turns)[1]["content"]
        # the blocks of a remembered reply are the SDK's own, the others are dicts
        return [getattr(block, "type", None) or block["type"] for block in content]

    assert replay("toolu_2") == ["thinking", "tool_use"]
    assert replay("toolu_1") == ["tool_use"]
    assert len(requests) == 2


async def test_reasoning_effort_is_sent_as_the_effort_of_the_output_config():
    sdk, requests = anthropic_client(message_reply("x"))
    llm = MessagesAdapter("m", client=sdk, model_options={"reasoning_effort": "low"})

    await llm.complete("s", "u")

    assert requests[0]["output_config"] == {"effort": "low"}


async def test_without_a_reasoning_effort_there_is_no_output_config():
    sdk, requests = anthropic_client(message_reply("x"))

    await MessagesAdapter("m", client=sdk).complete("s", "u")

    assert "output_config" not in requests[0]


async def test_extra_may_set_max_tokens_and_the_rest_is_sent_as_it_is():
    sdk, requests = anthropic_client(message_reply("x"))
    llm = MessagesAdapter(
        "m",
        client=sdk,
        model_options={"extra": {"max_tokens": 4000, "service_tier": "auto"}},
    )

    await llm.complete("s", "u")

    assert requests[0]["max_tokens"] == 4000
    assert requests[0]["service_tier"] == "auto"


def test_a_temperature_is_refused_and_none_is_not():
    sdk, _ = anthropic_client(message_reply("x"))

    MessagesAdapter("m", client=sdk, model_options={"temperature": None})
    with pytest.raises(ValueError, match="temperature cannot be set"):
        MessagesAdapter("m", client=sdk, model_options={"temperature": 0})


@pytest.mark.parametrize("field", ["system", "top_p", "top_k"])
def test_extra_may_not_set_what_the_client_sets_or_the_models_take_none_of(field: str):
    sdk, _ = anthropic_client(message_reply("x"))

    with pytest.raises(ValueError, match=f"extra cannot set {field}"):
        MessagesAdapter("m", client=sdk, model_options={"extra": {field: 1}})


def test_an_unknown_option_is_refused():
    sdk, _ = anthropic_client(message_reply("x"))

    mistyped = {"seed": 1}

    with pytest.raises(ValueError, match="model_options has no seed"):
        MessagesAdapter("m", client=sdk, model_options=mistyped)  # ty: ignore[invalid-argument-type]


async def test_the_timeout_of_the_defaults_and_the_one_given_are_the_ones_sent():
    seen: list[float | None] = []
    sdk, _ = anthropic_client(message_reply("x"), timeouts=seen)

    await MessagesAdapter("m", client=sdk).complete("s", "u")
    await MessagesAdapter("m", client=sdk, model_options={"timeout": 5}).complete(
        "s", "u"
    )

    assert seen == [120.0, 5.0]


async def test_the_retries_of_the_options_are_the_ones_the_sdk_makes():
    sdk, requests = anthropic_client(message_reply("x"), failures=2)
    llm = MessagesAdapter("m", client=sdk, model_options={"max_retries": 0})

    with pytest.raises(anthropic.InternalServerError):
        await llm.complete("s", "u")

    assert len(requests) == 1


async def test_with_model_asks_another_model_with_the_same_settings_and_no_memory():
    sdk, requests = anthropic_client(
        message_reply("x"),
        replies=[tool_use_reply([("toolu_1", "lookup", {})], thinking="one")],
    )
    llm = MessagesAdapter("big", client=sdk, model_options={"reasoning_effort": "low"})
    await llm.chat("s", [Message("user", "q")], [LOOKUP])

    cheap = llm.with_model("small", {"reasoning_effort": None})
    await cheap.complete("s", "u")

    assert [r["model"] for r in requests] == ["big", "small"]
    assert "output_config" not in requests[1]
    assert cheap._remembered == {}
    assert len(llm._remembered) == 1


def test_with_model_refuses_a_temperature_too():
    sdk, _ = anthropic_client(message_reply("x"))

    with pytest.raises(ValueError, match="temperature cannot be set"):
        MessagesAdapter("m", client=sdk).with_model("n", {"temperature": 0})


def test_tools_are_offered_unless_the_options_say_the_server_cannot_take_them():
    sdk, _ = anthropic_client(message_reply("x"))

    assert MessagesAdapter("m", client=sdk).supports_tools is True
    assert (
        MessagesAdapter("m", client=sdk, model_options={"tools": False}).supports_tools
        is False
    )
