import json
from typing import Any

import pytest
from support import chat_reply, chat_reply_ending, openai_client, tool_calls_reply

from text2cypher import (
    Completion,
    FakeLLM,
    Message,
    OpenAIClient,
    Tool,
    ToolCall,
    Usage,
)
from text2cypher.llm._base import supports_tools

pytestmark = pytest.mark.anyio

READ_SKILL = Tool(
    "read_skill",
    "Read one skill.",
    {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "The skill id."}},
        "required": ["id"],
    },
)


async def test_the_tools_are_offered_as_functions_with_a_json_schema():
    sdk, requests = openai_client(chat_reply("ok", None))

    await OpenAIClient("m", client=sdk).chat(
        "system", [Message("user", "q")], [READ_SKILL]
    )

    assert requests[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": "Read one skill.",
                "parameters": READ_SKILL.parameters,
            },
        }
    ]
    assert requests[0]["tool_choice"] == "auto"


async def test_a_call_that_forbids_tools_says_so_and_keeps_them_offered():
    sdk, requests = openai_client(chat_reply("ok", None))

    await OpenAIClient("m", client=sdk).chat(
        "s", [Message("user", "q")], [READ_SKILL], tool_choice="none"
    )

    assert requests[0]["tool_choice"] == "none"
    assert len(requests[0]["tools"]) == 1


async def test_without_tools_neither_tools_nor_a_choice_is_sent():
    sdk, requests = openai_client(chat_reply("ok", None))

    await OpenAIClient("m", client=sdk).chat("s", [Message("user", "q")])
    await OpenAIClient("m", client=sdk).chat("s", [Message("user", "q")], [])

    for request in requests:
        assert "tools" not in request
        assert "tool_choice" not in request


async def test_the_model_asking_for_tools_comes_back_as_tool_calls_with_parsed_arguments():
    usage = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    sdk, _ = openai_client(
        tool_calls_reply(
            [
                ("call_1", "read_skill", '{"id": "path-finding"}'),
                ("call_2", "read_skill", '{"id": "functions"}'),
            ],
            usage,
        )
    )

    completion = await OpenAIClient("m", client=sdk).chat(
        "s", [Message("user", "q")], [READ_SKILL]
    )

    assert completion == Completion(
        "",
        Usage(20, 4),
        (
            ToolCall("call_1", "read_skill", {"id": "path-finding"}),
            ToolCall("call_2", "read_skill", {"id": "functions"}),
        ),
        "tool_calls",
    )


@pytest.mark.parametrize("arguments", ["not json", "[1, 2]", '"text"', ""])
async def test_arguments_that_are_not_a_json_object_are_empty_so_the_model_can_be_told(
    arguments: str,
):
    sdk, _ = openai_client(tool_calls_reply([("c", "read_skill", arguments)]))

    completion = await OpenAIClient("m", client=sdk).chat(
        "s", [Message("user", "q")], [READ_SKILL]
    )

    assert completion.tool_calls == (ToolCall("c", "read_skill", {}),)


async def test_a_call_of_any_kind_but_a_function_is_ignored():
    reply = tool_calls_reply([("c", "x", "{}")], kind="custom")
    sdk, _ = openai_client(reply)

    completion = await OpenAIClient("m", client=sdk).chat(
        "s", [Message("user", "q")], [READ_SKILL]
    )

    assert completion.tool_calls == ()


async def test_the_turns_of_a_tool_exchange_are_sent_in_the_shape_the_api_wants():
    sdk, requests = openai_client(chat_reply("```cypher\nRETURN 1\n```", None))
    call = ToolCall("call_1", "read_skill", {"id": "functions"})

    await OpenAIClient("m", client=sdk).chat(
        "system",
        [
            Message("user", "q"),
            Message("assistant", "", (call,)),
            Message("tool", "# Functions\n...", tool_call_id="call_1"),
        ],
        [READ_SKILL],
        tool_choice="none",
    )

    assert requests[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_skill",
                        "arguments": json.dumps({"id": "functions"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "# Functions\n..."},
    ]


async def test_a_model_without_a_system_role_still_gets_the_tool_turns():
    sdk, requests = openai_client(chat_reply("ok", None))
    call = ToolCall("c", "read_skill", {"id": "x"})

    await OpenAIClient("m", client=sdk, system_role=False).chat(
        "system",
        [
            Message("user", "q"),
            Message("assistant", "", (call,)),
            Message("tool", "r", tool_call_id="c"),
        ],
        [READ_SKILL],
    )

    roles = [m["role"] for m in requests[0]["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert requests[0]["messages"][0]["content"] == "system\n\nq"


async def test_a_reply_with_text_and_no_tool_calls_is_as_before():
    sdk, _ = openai_client(chat_reply("```cypher\nRETURN 1\n```", None))

    completion = await OpenAIClient("m", client=sdk).chat(
        "s", [Message("user", "q")], [READ_SKILL]
    )

    assert completion == Completion("```cypher\nRETURN 1\n```", None, (), "stop")


def test_a_message_and_a_completion_without_tools_are_what_they_were():
    assert Message("user", "q") == Message("user", "q", (), None)
    assert Completion("a", None) == Completion("a", None, ())
    assert Completion("a", None).tool_calls == ()


class WithTools:
    async def chat(self, system, messages, tools=None, *, tool_choice="auto"):
        return Completion("", None)


class WithKwargs:
    async def chat(self, system, messages, **options: Any):
        return Completion("", None)


class TwoArguments:
    async def chat(self, system, messages):
        return Completion("", None)


class OnlyComplete:
    async def complete(self, system, user):
        return Completion("", None)


class Declares(TwoArguments):
    supports_tools = True


class Refuses(WithTools):
    supports_tools = False


@pytest.mark.parametrize(
    ("llm", "expected"),
    [
        (WithTools(), True),
        (WithKwargs(), True),
        (TwoArguments(), False),
        (OnlyComplete(), False),
        (Declares(), True),
        (Refuses(), False),
        (FakeLLM([]), True),
        (FakeLLM([], tools=False), False),
    ],
)
def test_a_client_can_be_offered_tools_if_it_says_so_or_takes_them(
    llm: object, expected: bool
):
    assert supports_tools(llm) is expected


def test_the_openai_client_can_say_it_cannot_be_offered_tools():
    sdk, _ = openai_client(chat_reply("x", None))

    assert supports_tools(OpenAIClient("m", client=sdk)) is True
    assert supports_tools(OpenAIClient("m", client=sdk, tools=False)) is False


async def test_the_fake_records_the_tools_and_the_choice_and_can_script_a_tool_call():
    call = ToolCall("c", "read_skill", {"id": "functions"})
    llm = FakeLLM([Completion("", Usage(3, 1), (call,)), "```cypher\nRETURN 1\n```"])

    first = await llm.chat("s", [Message("user", "q")], [READ_SKILL])
    second = await llm.chat(
        "s", [Message("user", "q")], [READ_SKILL], tool_choice="none"
    )

    assert first.tool_calls == (call,)
    assert second.tool_calls == ()
    assert [c.tools for c in llm.calls] == [(READ_SKILL,), (READ_SKILL,)]
    assert [c.tool_choice for c in llm.calls] == ["auto", "none"]


@pytest.mark.parametrize("reason", ["stop", "length", "content_filter"])
async def test_the_reason_a_reply_ended_comes_back_with_it(reason: str):
    sdk, _ = openai_client(chat_reply_ending("some text", reason))

    completion = await OpenAIClient("m", client=sdk).chat("s", [Message("user", "q")])

    assert completion.finish_reason == reason
    assert completion.text == "some text"


async def test_a_refusal_with_no_text_is_named_as_one():
    sdk, _ = openai_client(
        chat_reply_ending(None, "stop", refusal="I cannot help with that.")
    )

    completion = await OpenAIClient("m", client=sdk).chat("s", [Message("user", "q")])

    assert completion.finish_reason == "refusal"
    assert completion.text == ""


async def test_a_refusal_next_to_text_leaves_the_reason_as_the_api_gave_it():
    sdk, _ = openai_client(
        chat_reply_ending("```cypher\nRETURN 1\n```", "stop", refusal="partly")
    )

    completion = await OpenAIClient("m", client=sdk).chat("s", [Message("user", "q")])

    assert completion.finish_reason == "stop"


def test_a_completion_does_not_say_why_unless_it_is_told():
    assert Completion("a", None).finish_reason is None
    assert Completion("a", None, (), "length").finish_reason == "length"
