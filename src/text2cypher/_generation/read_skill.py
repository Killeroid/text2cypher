"""Let the model read skills, by calling a tool, while it writes a query."""

from collections.abc import Sequence

from text2cypher._results import NO_TOKENS
from text2cypher._skills import skills
from text2cypher.llm import AsyncChatClient, Completion, Message, Tool, ToolCall, Usage

READ_SKILL = Tool(
    "read_skill",
    "Read one skill: what it is for, the syntax, and an example that runs on FalkorDB.",
    {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The id of a skill from the list."}
        },
        "required": ["id"],
    },
)

# Rounds of tool calls before the model must answer, and the calls answered in one round
_ROUNDS = 3
_CALLS_PER_ROUND = 4


class SkillLoopFailed(Exception):
    """A call failed in the middle of reading skills; ``__cause__`` is why.

    Attributes:
        usage: The tokens of the calls that had succeeded, or ``None`` if unknown.
        calls: How many calls had succeeded.
    """

    def __init__(self, usage: Usage | None, calls: int) -> None:
        super().__init__("a call failed while reading skills")
        self.usage = usage
        self.calls = calls


async def chat_reading_skills(
    llm: AsyncChatClient, system: str, messages: Sequence[Message]
) -> tuple[Completion, tuple[str, ...], Usage | None]:
    """The model's reply, the ids of the skills it read, and the tokens of every call.

    The model is offered ``read_skill`` for up to three rounds; each call is answered with
    the skill (or what went wrong), and it is asked again. Then it must answer: the tool is
    still offered but a call is forbidden.

    Raises:
        SkillLoopFailed: If a call fails, with what the earlier calls cost.
    """
    turns = list(messages)
    read: list[str] = []
    usage: Usage | None = NO_TOKENS
    calls = 0
    try:
        for _ in range(_ROUNDS):
            completion = await llm.chat(system, turns, [READ_SKILL])
            usage, calls = _plus(usage, completion.usage), calls + 1
            if not completion.tool_calls:
                return completion, tuple(read), usage
            turns.append(Message("assistant", completion.text, completion.tool_calls))
            for number, call in enumerate(completion.tool_calls):
                content = (
                    _too_many if number >= _CALLS_PER_ROUND else _answer(call, read)
                )
                turns.append(Message("tool", content, tool_call_id=call.id))
        completion = await llm.chat(system, turns, [READ_SKILL], tool_choice="none")
        usage = _plus(usage, completion.usage)
        return completion, tuple(read), usage
    except Exception as error:
        raise SkillLoopFailed(usage, calls) from error


_too_many = "Too many skills at once: ask for fewer."


def _answer(call: ToolCall, read: list[str]) -> str:
    """What the model is told about one call to ``read_skill``; the id is added to ``read``."""
    available = skills()
    wanted = call.arguments.get("id")
    ids = ", ".join(available)
    if not isinstance(wanted, str) or not wanted:
        return f"Give the id of a skill. The ids are: {ids}."
    if wanted not in available:
        return f"There is no skill with the id `{wanted}`. The ids are: {ids}."
    if wanted not in read:
        read.append(wanted)
    return available[wanted].text


def _plus(total: Usage | None, used: Usage | None) -> Usage | None:
    return None if total is None or used is None else total + used
