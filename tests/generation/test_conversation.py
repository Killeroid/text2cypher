from cypher_guardrail import issues as cg
from support import FILM_SCHEMA, make_issue

from text2cypher._generation import Round
from text2cypher._generation.conversation import (
    CLOSING,
    SCHEMA_NOTE,
    chat_messages,
    folded_message,
)
from text2cypher.llm import Message

SCHEMA = FILM_SCHEMA
PERSN = Round(
    "MATCH (p:Persn) RETURN p.name",
    (
        make_issue(
            cg.UNKNOWN_LABEL, "The label Persn is not in the graph.", label="Persn"
        ),
    ),
)
MOVIE = Round(
    "MATCH (m:Movie {name: 'Heat'}) RETURN m",
    (
        make_issue(
            cg.INVALID_PROPERTY_ACCESS,
            "The property name is not on Movie.",
            kind="node",
            name="Movie",
            properties=["name"],
        ),
    ),
)


def test_the_first_turn_is_the_question_and_nothing_else_without_rounds():
    assert chat_messages("Who acted in Heat?", [], SCHEMA) == [
        Message("user", "# Question\nWho acted in Heat?")
    ]


def test_a_correction_is_the_query_the_llm_wrote_and_then_its_problems():
    messages = chat_messages("Who acted in Heat?", [PERSN], SCHEMA)

    assert messages == [
        Message("user", "# Question\nWho acted in Heat?"),
        Message("assistant", "```cypher\nMATCH (p:Persn) RETURN p.name\n```"),
        Message(
            "user",
            "This query has problems:\n\n"
            "- There is no label `Persn` in the schema.\n"
            "  Did you mean `Person`?\n\n"
            "Write a corrected query that fixes the problems and keeps what already "
            "works. Reply with the query in one ```cypher code block and nothing "
            "else. If the question cannot be answered with this schema, reply with "
            "an empty code block.",
        ),
    ]


def test_the_failed_query_is_not_quoted_again_in_the_turn_about_it():
    *_, last = chat_messages("q", [PERSN], SCHEMA)

    assert "MATCH" not in last.content


def test_every_round_adds_a_query_and_a_turn_in_order():
    messages = chat_messages("q", [PERSN, MOVIE], SCHEMA)

    assert [m.role for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "Persn" in messages[1].content
    assert "no label `Persn`" in messages[2].content
    assert "Movie {name" in messages[3].content
    assert "`Movie` has no property `name`" in messages[4].content
    assert messages[-1].content.endswith(CLOSING)


def test_a_repeated_problem_is_listed_once():
    same = make_issue(cg.UNKNOWN_LABEL, "The label X is not in the graph.", label="X")

    *_, last = chat_messages("q", [Round("MATCH (x:X) RETURN x", (same, same))], SCHEMA)

    assert last.content.count("no label `X`") == 1


def test_the_note_about_the_schema_opens_the_turn_of_the_round_that_has_it():
    noted = Round(PERSN.query, PERSN.issues, schema_note=True)

    messages = chat_messages("q", [noted, MOVIE], SCHEMA)

    assert messages[2].content.startswith(f"{SCHEMA_NOTE}\n\nThis query has problems:")
    assert SCHEMA_NOTE not in messages[4].content


def test_the_same_conversation_as_one_message_lists_the_attempts():
    message = folded_message("Who acted in Heat?", [PERSN, MOVIE], SCHEMA)

    assert message == (
        "# Question\nWho acted in Heat?\n\n"
        "# Earlier attempts, oldest first\n\n"
        "## Attempt 1\n"
        "```cypher\nMATCH (p:Persn) RETURN p.name\n```\n"
        "This query has problems:\n\n"
        "- There is no label `Persn` in the schema.\n"
        "  Did you mean `Person`?\n\n"
        "## Attempt 2\n"
        "```cypher\nMATCH (m:Movie {name: 'Heat'}) RETURN m\n```\n"
        "This query has problems:\n\n"
        "- The label `Movie` has no property `name`.\n"
        "  `name` is a property of the label `Person`.\n"
        f"  `Movie` has: title.\n\n{CLOSING}"
    )


def test_the_note_goes_before_the_attempts_in_one_message():
    noted = Round(PERSN.query, PERSN.issues, schema_note=True)

    message = folded_message("q", [noted], SCHEMA)

    assert message.startswith(
        f"# Question\nq\n\n# Earlier attempts, oldest first\n\n{SCHEMA_NOTE}\n\n## Attempt 1"
    )


def test_without_rounds_one_message_is_the_question():
    assert folded_message("q", [], SCHEMA) == "# Question\nq"


def test_a_query_that_failed_when_it_ran_is_followed_by_the_error_and_a_closing_for_it():
    failed = Round("MATCH (m:Movie) RETURN m.year % 0", (), "Division by zero")

    messages = chat_messages("How old?", [failed], SCHEMA)

    assert messages == [
        Message("user", "# Question\nHow old?"),
        Message("assistant", "```cypher\nMATCH (m:Movie) RETURN m.year % 0\n```"),
        Message(
            "user",
            "FalkorDB accepted this query but it failed when it ran:\n\n"
            "Division by zero\n\n"
            "Write a corrected query that fixes this and keeps what already works. "
            "Reply with the query in one ```cypher code block and nothing else. "
            "If the question cannot be answered with this schema, reply with an empty "
            "code block.",
        ),
    ]


def test_each_kind_of_failure_gets_its_own_turn_in_a_conversation_of_both():
    ran = Round("MATCH (m:Movie) RETURN m.year % 0", (), "Division by zero")

    messages = chat_messages("q", [PERSN, ran], SCHEMA)

    assert messages[2].content.startswith("This query has problems:")
    assert "fixes the problems" in messages[2].content
    assert messages[4].content.startswith("FalkorDB accepted this query")
    assert "fixes this and" in messages[4].content


def test_a_failure_to_run_is_listed_with_the_attempts_in_one_message():
    ran = Round("MATCH (m:Movie) RETURN m.year % 0", (), "Division by zero")

    message = folded_message("q", [PERSN, ran], SCHEMA)

    assert (
        "Attempt 2\n```cypher\nMATCH (m:Movie) RETURN m.year % 0\n```\n"
        "FalkorDB accepted this query but it failed when it ran:\n\n"
        f"Division by zero\n\n{CLOSING}"
    ) in message
