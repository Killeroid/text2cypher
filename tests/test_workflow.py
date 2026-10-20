import pytest
from support import FILM_SCHEMA, VALID, ScriptedValidator, invalid, reply

from text2cypher._config import Config
from text2cypher._generation import generate_cypher
from text2cypher._generation.prompts import (
    corrector_user,
    generator_system,
    generator_user,
)
from text2cypher._workflow import run_workflow
from text2cypher.llm import Completion, FakeLLM, Usage
from text2cypher.schema import GraphSchema

pytestmark = pytest.mark.anyio

FULL = FILM_SCHEMA
FILTERED = GraphSchema({"Person": {"name": ("String",)}}, {}, ())


async def test_a_valid_first_query_is_the_result():
    llm = FakeLLM([Completion(reply("MATCH (p:Person) RETURN p.name"), Usage(10, 4))])
    validate = ScriptedValidator(VALID)

    result = await run_workflow(
        "Who is there?", FULL, llm=llm, validate=validate, config=Config()
    )

    assert result.status == "valid"
    assert result.query == "MATCH (p:Person) RETURN p.name"
    assert [(a.query, a.schema_scope) for a in result.attempts] == [
        ("MATCH (p:Person) RETURN p.name", "full")
    ]
    assert result.attempts[0].validation == VALID
    assert result.usage == Usage(10, 4)
    assert [(c.system, c.user) for c in llm.calls] == [
        (generator_system(FULL), generator_user("Who is there?"))
    ]


async def test_a_query_with_problems_is_corrected_with_them_in_the_message():
    problem = invalid("The label Persn is not in the graph.")
    llm = FakeLLM(
        [reply("MATCH (p:Persn) RETURN p"), reply("MATCH (p:Person) RETURN p")]
    )
    validate = ScriptedValidator(problem, VALID)

    result = await run_workflow(
        "Who?", FULL, llm=llm, validate=validate, config=Config()
    )

    assert result.status == "valid"
    assert [a.query for a in result.attempts] == [
        "MATCH (p:Persn) RETURN p",
        "MATCH (p:Person) RETURN p",
    ]
    assert result.attempts[0].validation == problem
    assert validate.queries == ["MATCH (p:Persn) RETURN p", "MATCH (p:Person) RETURN p"]
    assert llm.calls[1] == (
        generator_system(FULL),
        corrector_user("Who?", "MATCH (p:Persn) RETURN p", problem.issues, FULL),
    )


async def test_it_fails_when_the_corrections_run_out():
    llm = FakeLLM([reply("a"), reply("b"), reply("c"), reply("never asked")])
    validate = ScriptedValidator(invalid("x"), invalid("y"), invalid("z"))

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=1, total_attempts=2),
    )

    # the first query and two corrections
    assert result.status == "failed"
    assert [a.query for a in result.attempts] == ["a", "b", "c"]
    assert result.query == "c"
    assert len(llm.calls) == 3


async def test_with_no_corrections_allowed_the_first_invalid_query_fails():
    llm = FakeLLM([reply("a")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x")),
        config=Config(attempts_w_filtered=0, total_attempts=0),
    )

    assert result.status == "failed"
    assert len(llm.calls) == 1


async def test_it_escalates_to_the_full_schema_after_the_filtered_attempts():
    llm = FakeLLM([reply(q) for q in "abcd"])
    validate = ScriptedValidator(*(invalid(q) for q in "abc"), VALID)

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=2, total_attempts=3),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.status == "valid"
    # the first query and two corrections use the filtered schema, the next one the full
    assert [a.schema_scope for a in result.attempts] == [
        "filtered",
        "filtered",
        "filtered",
        "full",
    ]
    assert [c.system for c in llm.calls] == [
        generator_system(s) for s in (FILTERED, FILTERED, FILTERED, FULL)
    ]


async def test_the_filter_is_given_the_full_schema_and_the_question():
    seen = []

    def schema_filter(schema: GraphSchema, question: str) -> GraphSchema:
        seen.append((schema, question))
        return FILTERED

    await run_workflow(
        "Who acted?",
        FULL,
        llm=FakeLLM([reply("a")]),
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=schema_filter,
    )

    assert seen == [(FULL, "Who acted?")]


async def test_without_a_filter_every_attempt_uses_the_full_schema():
    llm = FakeLLM([reply("a"), reply("b")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )

    assert [a.schema_scope for a in result.attempts] == ["full", "full"]
    assert {c.system for c in llm.calls} == {generator_system(FULL)}


async def test_an_empty_first_reply_is_unanswerable_and_nothing_is_validated():
    llm = FakeLLM([Completion("```cypher\n```", Usage(9, 1))])
    validate = ScriptedValidator()

    result = await run_workflow(
        "Delete everyone", FULL, llm=llm, validate=validate, config=Config()
    )

    assert result.status == "unanswerable"
    assert result.query == ""
    assert validate.queries == []
    assert len(llm.calls) == 1
    assert [(a.query, a.validation) for a in result.attempts] == [("", None)]
    assert result.usage == Usage(9, 1)


async def test_an_empty_reply_to_a_correction_is_unanswerable_too():
    llm = FakeLLM([reply("MATCH (x) RETURN x"), "```cypher\n```"])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x")),
        config=Config(),
    )

    assert result.status == "unanswerable"
    assert [a.query for a in result.attempts] == ["MATCH (x) RETURN x", ""]
    assert result.query == ""


async def test_usage_is_the_sum_only_when_every_attempt_reported_it():
    first = Completion(reply("a"), Usage(10, 2))
    second = Completion(reply("b"), Usage(5, 1))
    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([first, second]),
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )
    assert result.usage == Usage(15, 3)

    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([first, reply("b")]),
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )
    assert result.usage is None


async def test_the_workflow_asks_for_the_first_query_like_generate_does():
    llm = FakeLLM([reply("a")])
    await run_workflow(
        "Q", FULL, llm=llm, validate=ScriptedValidator(VALID), config=Config()
    )
    direct = FakeLLM([reply("a")])
    await generate_cypher(direct, FULL, "Q")

    assert llm.calls == direct.calls


async def test_a_filter_that_keeps_everything_is_not_a_filter():
    llm = FakeLLM([reply("a"), reply("b")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(attempts_w_filtered=1, total_attempts=2),
        schema_filter=lambda schema, question: schema,
    )

    assert [a.schema_scope for a in result.attempts] == ["full", "full"]
    assert {c.system for c in llm.calls} == {generator_system(FULL)}


async def test_declining_on_the_filtered_schema_is_asked_again_with_the_full_one():
    llm = FakeLLM(["```cypher\n```", reply("MATCH (m:Movie) RETURN m.title")])
    validate = ScriptedValidator(VALID)

    result = await run_workflow(
        "Which movies?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.status == "valid"
    assert [(a.query, a.schema_scope) for a in result.attempts] == [
        ("", "filtered"),
        ("MATCH (m:Movie) RETURN m.title", "full"),
    ]
    assert [(c.system, c.user) for c in llm.calls] == [
        (generator_system(FILTERED), generator_user("Which movies?")),
        (generator_system(FULL), generator_user("Which movies?")),
    ]


async def test_declining_on_the_full_schema_after_the_filtered_one_is_unanswerable():
    llm = FakeLLM(["```cypher\n```", "```cypher\n```"])

    result = await run_workflow(
        "What is the weather?",
        FULL,
        llm=llm,
        validate=ScriptedValidator(),
        config=Config(),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.status == "unanswerable"
    assert [a.schema_scope for a in result.attempts] == ["filtered", "full"]
    assert len(llm.calls) == 2


async def test_declining_a_correction_on_the_filtered_schema_asks_for_it_on_the_full_one():
    problem = invalid("The label X is not in the graph.")
    llm = FakeLLM([reply("MATCH (x:X) RETURN x"), "```cypher\n```", reply("good")])
    validate = ScriptedValidator(problem, VALID)

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=3, total_attempts=3),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.status == "valid"
    assert [a.schema_scope for a in result.attempts] == ["filtered", "filtered", "full"]
    assert [(c.system, c.user) for c in llm.calls[1:]] == [
        (
            generator_system(FILTERED),
            corrector_user("Q", "MATCH (x:X) RETURN x", problem.issues, FILTERED),
        ),
        (
            generator_system(FULL),
            corrector_user("Q", "MATCH (x:X) RETURN x", problem.issues, FULL),
        ),
    ]
    # it was not a correction of its own: the query was corrected once
    assert validate.queries == ["MATCH (x:X) RETURN x", "good"]


async def test_after_declining_on_the_filtered_schema_later_corrections_use_the_full_one():
    llm = FakeLLM(["```cypher\n```", reply("a"), reply("b")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(attempts_w_filtered=5, total_attempts=5),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert [a.schema_scope for a in result.attempts] == ["filtered", "full", "full"]
    assert {c.system for c in llm.calls[1:]} == {generator_system(FULL)}
