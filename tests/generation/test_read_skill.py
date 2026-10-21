import pytest
from support import FILM_SCHEMA, reply

from text2cypher import Completion, FakeLLM, Message, ToolCall, Usage
from text2cypher._generation import Generation, Round, SkillLoopFailed, generate_cypher
from text2cypher._generation.prompts import SkillsOffer, generator_system
from text2cypher._generation.read_skill import READ_SKILL
from text2cypher._skills import skills

pytestmark = pytest.mark.anyio

SCHEMA = FILM_SCHEMA
QUERY = "MATCH (p:Person) RETURN p.name"
READ_USAGE = Usage(20, 5)
ANSWER_USAGE = Usage(10, 4)


def read(
    skill_id: str, call_id: str = "c1", usage: Usage | None = READ_USAGE
) -> Completion:
    return Completion("", usage, (ToolCall(call_id, "read_skill", {"id": skill_id}),))


def answer(usage: Usage | None = ANSWER_USAGE) -> Completion:
    return Completion(reply(QUERY), usage)


def test_the_tool_is_the_one_the_prompts_describe():
    assert READ_SKILL.name == "read_skill"
    assert READ_SKILL.description == (
        "Read one skill: what it is for, the syntax, and an example that runs on FalkorDB."
    )
    assert READ_SKILL.parameters == {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The id of a skill from the list."}
        },
        "required": ["id"],
    }


async def test_the_model_reads_a_skill_and_is_asked_again_with_what_it_read():
    llm = FakeLLM([read("path-finding"), answer()])

    generation = await generate_cypher(llm, SCHEMA, "Route?", skills=True)

    assert generation == Generation(QUERY, Usage(30, 9), ("path-finding",))
    first, second = llm.calls
    assert first.tools == (READ_SKILL,)
    assert first.tool_choice == "auto"
    call = ToolCall("c1", "read_skill", {"id": "path-finding"})
    assert second.messages == (
        Message("user", "Question: Route?"),
        Message("assistant", "", (call,)),
        Message("tool", skills()["path-finding"].text, tool_call_id="c1"),
    )
    assert second.tools == (READ_SKILL,)
    assert second.system == first.system


async def test_every_call_of_a_round_is_answered_with_its_own_id():
    both = Completion(
        "",
        None,
        (
            ToolCall("a", "read_skill", {"id": "path-finding"}),
            ToolCall("b", "read_skill", {"id": "functions"}),
        ),
    )
    llm = FakeLLM([both, answer(None)])

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    tool_turns = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in tool_turns] == [
        ("a", skills()["path-finding"].text),
        ("b", skills()["functions"].text),
    ]
    assert generation.skills == ("path-finding", "functions")
    # a call that reported no tokens makes the total unknown
    assert generation.usage is None


async def test_a_skill_read_twice_is_listed_once():
    llm = FakeLLM([read("functions", "a"), read("functions", "b"), answer()])

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert generation.skills == ("functions",)


@pytest.mark.parametrize(
    ("arguments", "told"),
    [
        (
            {"id": "nope"},
            "There is no skill with the id `nope`. The ids are: functions, graph-algorithms, path-finding.",
        ),
        (
            {},
            "Give the id of a skill. The ids are: functions, graph-algorithms, path-finding.",
        ),
        (
            {"id": 7},
            "Give the id of a skill. The ids are: functions, graph-algorithms, path-finding.",
        ),
    ],
)
async def test_a_call_with_no_such_skill_is_told_which_there_are_and_reads_nothing(
    arguments: dict, told: str
):
    llm = FakeLLM(
        [Completion("", None, (ToolCall("c", "read_skill", arguments),)), answer()]
    )

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert llm.calls[1].messages[-1] == Message("tool", told, tool_call_id="c")
    assert generation.skills == ()


async def test_calls_beyond_four_in_one_round_are_answered_but_not_read():
    six = Completion(
        "",
        None,
        tuple(ToolCall(str(i), "read_skill", {"id": "functions"}) for i in range(6)),
    )
    llm = FakeLLM([six, answer()])

    await generate_cypher(llm, SCHEMA, "q", skills=True)

    tool_turns = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_turns] == ["0", "1", "2", "3", "4", "5"]
    assert [m.content for m in tool_turns[4:]] == [
        "Too many skills at once: ask for fewer."
    ] * 2


async def test_after_three_rounds_the_model_must_answer_with_the_tool_still_offered():
    llm = FakeLLM(
        [
            read("functions", "1"),
            read("path-finding", "2"),
            read("graph-algorithms", "3"),
            answer(),
        ]
    )

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert len(llm.calls) == 4
    assert [c.tool_choice for c in llm.calls] == ["auto", "auto", "auto", "none"]
    assert all(c.tools == (READ_SKILL,) for c in llm.calls)
    assert generation.skills == ("functions", "path-finding", "graph-algorithms")
    assert generation.query == QUERY


async def test_a_call_that_fails_after_others_succeeded_says_what_they_cost():
    llm = FakeLLM([read("functions"), ConnectionError("down")])

    with pytest.raises(SkillLoopFailed) as raised:
        await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert raised.value.usage == Usage(20, 5)
    assert raised.value.calls == 1
    assert isinstance(raised.value.__cause__, ConnectionError)


async def test_a_client_that_cannot_be_offered_tools_has_every_skill_in_the_prompt():
    llm = FakeLLM([answer()], tools=False)

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    [call] = llm.calls
    assert call.tools == ()
    for skill in skills().values():
        assert skill.text in call.system
    assert "read_skill" not in call.system
    assert generation == Generation(QUERY, Usage(10, 4), ())


async def test_a_client_with_only_complete_has_every_skill_in_the_prompt_too():
    class CompleteOnly:
        def __init__(self) -> None:
            self.asked: list[tuple[str, str]] = []

        async def complete(self, system: str, user: str) -> Completion:
            self.asked.append((system, user))
            return answer()

    llm = CompleteOnly()

    await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert all(skill.text in llm.asked[0][0] for skill in skills().values())


async def test_without_the_option_no_tool_is_offered_and_the_prompt_has_no_skills():
    llm = FakeLLM([answer()])

    await generate_cypher(llm, SCHEMA, "q")

    [call] = llm.calls
    assert call.tools == ()
    assert call.system == generator_system(SCHEMA)
    assert "read_skill" not in call.system


async def test_what_an_earlier_request_read_is_in_the_prompt_of_the_next_one():
    failed = Round(QUERY, (), skills_read=("path-finding",))
    llm = FakeLLM([answer()])

    await generate_cypher(llm, SCHEMA, "q", [failed], skills=True)

    [call] = llm.calls
    assert call.system == generator_system(
        SCHEMA, skills=SkillsOffer(tools=True, read=("path-finding",))
    )
    assert "Skills you have already read:\n\n# Path finding" in call.system
    # still offered: the ones not read yet
    assert call.tools == (READ_SKILL,)
    # and the conversation stays plain turns: the reading is not replayed
    assert [m.role for m in call.messages] == ["user", "assistant", "user"]


async def test_a_final_reply_with_no_text_after_reading_a_skill_is_an_empty_reply():
    llm = FakeLLM([read("functions"), Completion("", Usage(5, 0))])

    generation = await generate_cypher(llm, SCHEMA, "q", skills=True)

    assert generation.empty
    assert generation.skills == ("functions",)
    assert generation.usage == Usage(25, 5)
