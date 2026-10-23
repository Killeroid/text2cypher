"""The conversation with the LLM: the question, then each query it wrote and what was wrong."""

from collections.abc import Sequence
from dataclasses import dataclass

from text2cypher._generation.problems import problems
from text2cypher._generation.prompts import generator_user
from text2cypher._validation import Issue
from text2cypher.llm import Message
from text2cypher.schema import GraphSchema

_REPLY = (
    "Reply with the query in one ```cypher code block and nothing else. "
    "If the question cannot be answered with this schema, reply with an empty code block."
)
CLOSING = f"Write a corrected query that fixes the problems and keeps what already works. {_REPLY}"
_CLOSING_AFTER_ERROR = (
    f"Write a corrected query that fixes this and keeps what already works. {_REPLY}"
)
SCHEMA_NOTE = (
    "The schema in the system message now lists the whole graph; before, it showed "
    "only part of it."
)


@dataclass(frozen=True, slots=True)
class Round:
    """A query the LLM wrote that has to be corrected, and what was wrong with it.

    That is its ``issues``, or the ``execution_error`` FalkorDB gave when it ran a query
    that had passed validation.

    ``schema_note`` marks the round whose turn tells the LLM that the schema is now the
    whole graph, after queries written with a part of it. ``skills_read`` are the ids of the
    skills the LLM read to write the query; the next request already has them in its prompt.
    """

    query: str
    issues: tuple[Issue, ...]
    execution_error: str | None = None
    schema_note: bool = False
    skills_read: tuple[str, ...] = ()


def chat_messages(
    question: str,
    rounds: Sequence[Round],
    schema: GraphSchema,
    *,
    parameterize: bool = False,
) -> list[Message]:
    """The conversation as turns: the question, then for each round the query the LLM
    wrote and a user turn with its problems. The problems are worded with ``schema``,
    the one the next query is written for."""
    messages = [Message("user", generator_user(question))]
    for round_ in rounds:
        note = f"{SCHEMA_NOTE}\n\n" if round_.schema_note else ""
        messages.append(Message("assistant", _fenced(round_.query)))
        closing = CLOSING if round_.execution_error is None else _CLOSING_AFTER_ERROR
        messages.append(
            Message(
                "user",
                f"{note}{_feedback(round_, schema, parameterize)}\n\n{closing}",
            )
        )
    return messages


def folded_message(
    question: str,
    rounds: Sequence[Round],
    schema: GraphSchema,
    *,
    parameterize: bool = False,
) -> str:
    """The same conversation as one user message, for a client that cannot be given turns."""
    if not rounds:
        return generator_user(question)
    note = f"{SCHEMA_NOTE}\n\n" if any(r.schema_note for r in rounds) else ""
    attempts = "\n\n".join(
        f"## Attempt {number}\n{_fenced(r.query)}\n{_feedback(r, schema, parameterize)}"
        for number, r in enumerate(rounds, start=1)
    )
    return (
        f"{generator_user(question)}\n\n# Earlier attempts, oldest first\n\n"
        f"{note}{attempts}\n\n{CLOSING}"
    )


def _fenced(query: str) -> str:
    return f"```cypher\n{query}\n```"


def _feedback(round_: Round, schema: GraphSchema, parameterize: bool) -> str:
    if round_.execution_error is not None:
        return (
            "FalkorDB accepted this query but it failed when it ran:\n\n"
            f"{round_.execution_error}"
        )
    return "This query has problems:\n\n" + problems(
        round_.issues, schema, parameterized=parameterize
    )
