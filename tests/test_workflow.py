from dataclasses import replace

import pytest
from cypher_guardrail import issues as cg
from support import (
    FILM_SCHEMA,
    RAN,
    ROWS,
    VALID,
    ScriptedExecutor,
    ScriptedValidator,
    failed_to_run,
    invalid,
    misspelled,
    reply,
)

from text2cypher._config import Config
from text2cypher._execution import Execution
from text2cypher._generation import FilterResult, Round, generate_cypher
from text2cypher._generation.conversation import SCHEMA_NOTE, chat_messages
from text2cypher._generation.prompts import (
    SkillsOffer,
    generator_system,
    generator_user,
)
from text2cypher._generation.read_skill import READ_SKILL
from text2cypher._workflow import run_workflow
from text2cypher.llm import Completion, FakeLLM, ToolCall, Usage
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
    assert llm.calls[1].system == generator_system(FULL)
    assert list(llm.calls[1].messages) == chat_messages(
        "Who?", [Round("MATCH (p:Persn) RETURN p", problem.issues)], FULL
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
    failed = Round("MATCH (x:X) RETURN x", problem.issues)
    assert [(c.system, list(c.messages)) for c in llm.calls[1:]] == [
        (generator_system(FILTERED), chat_messages("Q", [failed], FILTERED)),
        (
            generator_system(FULL),
            chat_messages("Q", [replace(failed, schema_note=True)], FULL),
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


async def test_parameters_change_the_system_prompt_of_every_request():
    llm = FakeLLM([reply("a"), reply("b")])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(use_parameters=True),
    )

    assert [c.system for c in llm.calls] == [
        generator_system(FULL, parameterize=True)
    ] * 2
    assert list(llm.calls[1].messages) == chat_messages(
        "Q", [Round("a", invalid("x").issues)], FULL, parameterize=True
    )


async def test_each_correction_carries_every_earlier_query_and_its_problems():
    problems = [invalid(text) for text in ("x", "y", "z")]
    llm = FakeLLM([reply(q) for q in "abcd"])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(*problems, VALID),
        config=Config(),
    )

    rounds = [Round(q, p.issues) for q, p in zip("abc", problems, strict=True)]
    assert [list(c.messages) for c in llm.calls] == [
        chat_messages("Q", rounds[:n], FULL) for n in range(4)
    ]
    assert [m.role for m in llm.calls[3].messages] == ["user"] + [
        "assistant",
        "user",
    ] * 3


async def test_the_llm_is_told_once_that_the_schema_is_now_the_whole_graph():
    llm = FakeLLM([reply(q) for q in "abcde"])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(*(invalid(q) for q in "abcd"), VALID),
        config=Config(attempts_w_filtered=1, total_attempts=4),
        schema_filter=lambda schema, question: FILTERED,
    )

    # two requests with the filtered schema, then three with the full one
    users = [[m.content for m in c.messages if m.role == "user"] for c in llm.calls]
    assert [sum(SCHEMA_NOTE in text for text in turns) for turns in users] == [
        0,
        0,
        1,
        1,
        1,
    ]
    # it opens the problems of the last query written with the filtered schema
    assert users[2][-1].startswith(SCHEMA_NOTE)
    # and it stays where it was, in the turn it was first given
    assert users[4][2].startswith(SCHEMA_NOTE)
    assert not users[4][-1].startswith(SCHEMA_NOTE)


async def test_there_is_no_note_when_the_schema_never_changed():
    llm = FakeLLM([reply("a"), reply("b")])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )

    assert all(SCHEMA_NOTE not in m.content for c in llm.calls for m in c.messages)


async def test_there_is_no_note_when_the_first_query_was_declined_on_the_filtered_schema():
    llm = FakeLLM(["```cypher\n```", reply("a")])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=lambda schema, question: FILTERED,
    )

    # nothing was written with the filtered schema, so there is nothing to explain
    assert list(llm.calls[1].messages) == chat_messages("Q", [], FULL)


async def test_a_declined_reply_is_not_part_of_the_conversation_that_is_asked_again():
    problem = invalid("The label X is not in the graph.")
    llm = FakeLLM([reply("MATCH (x:X) RETURN x"), "```cypher\n```", reply("good")])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(problem, VALID),
        config=Config(attempts_w_filtered=3, total_attempts=3),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert len(llm.calls[2].messages) == len(llm.calls[1].messages)
    assert [m.role for m in llm.calls[2].messages] == ["user", "assistant", "user"]


async def test_a_query_that_passed_validation_is_run_and_its_rows_are_the_result():
    execute = ScriptedExecutor(RAN)

    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([reply("a")]),
        validate=ScriptedValidator(VALID),
        config=Config(),
        execute=execute,
    )

    assert result.status == "valid"
    assert result.records == ROWS
    assert result.execution_error is None
    assert execute.queries == ["a"]


async def test_a_query_is_not_run_when_it_did_not_pass_validation():
    execute = ScriptedExecutor(RAN)

    await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([reply("a"), reply("b")]),
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
        execute=execute,
    )

    assert execute.queries == ["b"]


async def test_a_query_that_fails_when_run_is_corrected_with_what_failed():
    llm = FakeLLM([reply("a"), reply("b")])
    execute = ScriptedExecutor(failed_to_run("Division by zero"), RAN)

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID, VALID),
        config=Config(),
        execute=execute,
    )

    assert result.status == "valid"
    assert result.records == ROWS
    assert [a.execution_error for a in result.attempts] == ["Division by zero", None]
    assert result.execution_error is None
    assert execute.queries == ["a", "b"]
    assert list(llm.calls[1].messages) == chat_messages(
        "Q", [Round("a", (), "Division by zero")], FULL
    )


async def test_a_failure_to_run_after_a_failure_to_validate_is_corrected_the_same_way():
    problem = invalid("The label X is not in the graph.")
    llm = FakeLLM([reply("a"), reply("b"), reply("c")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(problem, VALID, VALID),
        config=Config(),
        execute=ScriptedExecutor(failed_to_run("boom"), RAN),
    )

    assert result.status == "valid"
    assert list(llm.calls[2].messages) == chat_messages(
        "Q", [Round("a", problem.issues), Round("b", (), "boom")], FULL
    )


async def test_a_correction_after_a_failure_to_run_counts_like_any_other():
    llm = FakeLLM([reply("a"), reply("b"), reply("never asked")])
    execute = ScriptedExecutor(failed_to_run("one"), failed_to_run("two"))

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID, VALID),
        config=Config(attempts_w_filtered=1, total_attempts=1),
        execute=execute,
    )

    # when the corrections run out on queries FalkorDB fails, no query both validated and ran
    assert result.status == "failed"
    assert result.query == "b"
    assert result.records is None
    assert result.execution_error == "two"
    assert [a.execution_error for a in result.attempts] == ["one", "two"]
    assert len(llm.calls) == 2


async def test_a_timeout_or_a_lost_connection_ends_the_loop_with_the_valid_query():
    llm = FakeLLM([reply("a"), reply("never asked")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID),
        config=Config(),
        execute=ScriptedExecutor(Execution(error="Query timed out")),
    )

    assert result.status == "valid"
    assert result.query == "a"
    assert result.records is None
    assert result.execution_error == "Query timed out"
    assert len(llm.calls) == 1


async def test_nothing_is_run_without_an_executor():
    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([reply("a")]),
        validate=ScriptedValidator(VALID),
        config=Config(),
    )

    assert result.records is None
    assert result.execution_error is None


async def test_a_failure_to_run_on_the_filtered_schema_can_move_on_to_the_full_one():
    llm = FakeLLM([reply("a"), reply("b")])

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID, VALID),
        config=Config(attempts_w_filtered=0, total_attempts=2),
        schema_filter=lambda schema, question: FILTERED,
        execute=ScriptedExecutor(failed_to_run("boom"), RAN),
    )

    assert [c.system for c in llm.calls] == [
        generator_system(FILTERED),
        generator_system(FULL),
    ]
    assert llm.calls[1].messages[-1].content.startswith(SCHEMA_NOTE)


TYPO = "MATCH (p:Persn) RETURN p.name"
FIXED = "MATCH (p:Person) RETURN p.name"


async def test_a_typo_is_fixed_without_the_llm_and_costs_no_correction():
    llm = FakeLLM([Completion(reply(TYPO), Usage(10, 4))])
    validate = ScriptedValidator(misspelled(label="Persn"), VALID)

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=0, total_attempts=0, autofix_typos=True),
    )

    assert result.status == "valid"
    assert result.query == FIXED
    assert validate.queries == [TYPO, FIXED]
    assert len(llm.calls) == 1
    # the query the LLM wrote and the one that was fixed are both there, and the fix cost nothing
    assert [(a.query, a.auto_fixed) for a in result.attempts] == [
        (TYPO, False),
        (FIXED, True),
    ]
    assert result.attempts[0].validation == misspelled(label="Persn")
    assert result.attempts[1].validation == VALID
    assert result.attempts[1].usage == Usage(0, 0)
    assert result.usage == Usage(10, 4)


async def test_a_typo_is_not_fixed_unless_asked():
    llm = FakeLLM([reply(TYPO), reply(FIXED)])
    validate = ScriptedValidator(misspelled(label="Persn"), VALID)

    result = await run_workflow(
        "Who?", FULL, llm=llm, validate=validate, config=Config()
    )

    assert validate.queries == [TYPO, FIXED]
    assert len(llm.calls) == 2
    assert [a.auto_fixed for a in result.attempts] == [False, False]


async def test_a_fix_that_does_not_pass_is_dropped_and_the_llm_gets_the_query_it_wrote():
    llm = FakeLLM([reply(TYPO), reply("MATCH (p:Person) RETURN p")])
    validate = ScriptedValidator(
        misspelled(label="Persn"), invalid("still wrong"), VALID
    )

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(autofix_typos=True),
    )

    assert result.status == "valid"
    assert validate.queries == [TYPO, FIXED, "MATCH (p:Person) RETURN p"]
    assert [a.query for a in result.attempts] == [TYPO, "MATCH (p:Person) RETURN p"]
    assert not any(a.auto_fixed for a in result.attempts)
    # the correction is about the query the LLM wrote, with the problems it had
    assert list(llm.calls[1].messages) == chat_messages(
        "Who?", [Round(TYPO, misspelled(label="Persn").issues)], FULL
    )


async def test_a_problem_that_is_not_a_typo_goes_to_the_llm_without_a_fix_being_tried():
    llm = FakeLLM(
        [reply("MATCH (r:Robot) RETURN r"), reply("MATCH (p:Person) RETURN p")]
    )
    validate = ScriptedValidator(invalid("The label Robot is not in the graph."), VALID)

    await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(autofix_typos=True),
    )

    assert validate.queries == ["MATCH (r:Robot) RETURN r", "MATCH (p:Person) RETURN p"]


async def test_a_fixed_query_is_the_one_that_is_run_and_corrected_if_it_fails_to_run():
    llm = FakeLLM([reply(TYPO), reply("MATCH (p:Person) RETURN p.born")])
    execute = ScriptedExecutor(failed_to_run("boom"), RAN)

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=ScriptedValidator(misspelled(label="Persn"), VALID, VALID),
        config=Config(autofix_typos=True),
        execute=execute,
    )

    assert result.status == "valid"
    assert execute.queries == [FIXED, "MATCH (p:Person) RETURN p.born"]
    assert [a.execution_error for a in result.attempts] == [None, "boom", None]
    assert list(llm.calls[1].messages) == chat_messages(
        "Who?", [Round(FIXED, (), "boom")], FULL
    )


async def test_a_fix_that_shows_another_typo_is_followed_by_a_second_fix():
    # with the label wrong the properties could not be checked, so the second typo shows only now
    query = "MATCH (p:Persn) RETURN p.nme"
    llm = FakeLLM([reply(query)])
    validate = ScriptedValidator(
        misspelled(label="Persn"), misspelled(prop="nme"), VALID
    )

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=0, total_attempts=0, autofix_typos=True),
    )

    assert result.status == "valid"
    assert validate.queries == [
        query,
        "MATCH (p:Person) RETURN p.nme",
        "MATCH (p:Person) RETURN p.name",
    ]
    assert [(a.query, a.auto_fixed) for a in result.attempts] == [
        (query, False),
        ("MATCH (p:Person) RETURN p.name", True),
    ]
    assert len(llm.calls) == 1


async def test_the_fixes_are_given_up_after_three_rounds():
    query = "MATCH (p:Persn)-[:ACTD_IN]->(m:Movi) RETURN p.nme"
    llm = FakeLLM([reply(query)])
    # each fix shows another layer: the label, then the type, then the property, then one more label
    validate = ScriptedValidator(
        misspelled(label="Persn"),
        misspelled(type="ACTD_IN"),
        misspelled(prop="nme"),
        misspelled(label="Movi"),
    )

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=0, total_attempts=0, autofix_typos=True),
    )

    # the query as it is, and three fixes; the fourth would be one fix too many
    assert result.status == "failed"
    assert validate.queries == [
        query,
        "MATCH (p:Person)-[:ACTD_IN]->(m:Movi) RETURN p.nme",
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movi) RETURN p.nme",
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movi) RETURN p.name",
    ]
    # none of them passed, so none is kept: the result is the query the LLM wrote
    assert result.query == query
    assert not any(a.auto_fixed for a in result.attempts)


HEADER_BROKEN = "name='Ann', year=1990 MATCH (p:Person {name: $name}) RETURN p.name"
HEADER_FIXED = (
    "CYPHER name='Ann' year=1990 MATCH (p:Person {name: $name}) RETURN p.name"
)


async def test_a_broken_parameter_header_is_repaired_without_the_llm_and_costs_nothing():
    llm = FakeLLM([Completion(reply(HEADER_BROKEN), Usage(10, 4))])
    validate = ScriptedValidator(invalid("syntax", cg.SYNTAX_ERROR), VALID)

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=0, total_attempts=0, use_parameters=True),
    )

    assert result.status == "valid"
    assert result.query == HEADER_FIXED
    assert validate.queries == [HEADER_BROKEN, HEADER_FIXED]
    assert len(llm.calls) == 1
    assert [(a.query, a.auto_fixed) for a in result.attempts] == [
        (HEADER_BROKEN, False),
        (HEADER_FIXED, True),
    ]
    assert result.attempts[1].usage == Usage(0, 0)
    assert result.usage == Usage(10, 4)


async def test_the_header_is_not_repaired_when_the_values_are_not_in_one():
    llm = FakeLLM([reply(HEADER_BROKEN), reply(HEADER_FIXED)])
    validate = ScriptedValidator(invalid("syntax", cg.SYNTAX_ERROR), VALID)

    result = await run_workflow(
        "Who?", FULL, llm=llm, validate=validate, config=Config()
    )

    # a query that is not asked for with parameters is left to the LLM, as it was
    assert validate.queries == [HEADER_BROKEN, HEADER_FIXED]
    assert len(llm.calls) == 2
    assert not any(a.auto_fixed for a in result.attempts)


async def test_a_repair_that_does_not_pass_is_dropped():
    llm = FakeLLM([reply(HEADER_BROKEN), reply("MATCH (p:Person) RETURN p.name")])
    validate = ScriptedValidator(
        invalid("syntax", cg.SYNTAX_ERROR), invalid("still wrong"), VALID
    )

    result = await run_workflow(
        "Who?", FULL, llm=llm, validate=validate, config=Config(use_parameters=True)
    )

    assert result.status == "valid"
    assert [a.query for a in result.attempts] == [
        HEADER_BROKEN,
        "MATCH (p:Person) RETURN p.name",
    ]
    assert not any(a.auto_fixed for a in result.attempts)


async def test_the_header_is_repaired_first_and_then_the_typos_if_they_are_asked_for():
    broken = "name='Ann', year=1990 MATCH (p:Persn {name: $name}) RETURN p.name"
    llm = FakeLLM([reply(broken)])
    validate = ScriptedValidator(
        invalid("syntax", cg.SYNTAX_ERROR), misspelled(label="Persn"), VALID
    )

    result = await run_workflow(
        "Who?",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(
            attempts_w_filtered=0,
            total_attempts=0,
            use_parameters=True,
            autofix_typos=True,
        ),
    )

    assert result.status == "valid"
    assert validate.queries == [
        broken,
        "CYPHER name='Ann' year=1990 MATCH (p:Persn {name: $name}) RETURN p.name",
        HEADER_FIXED,
    ]
    assert len(llm.calls) == 1


async def test_a_filter_that_is_async_is_awaited_and_what_it_cost_is_on_the_result():
    seen = []

    async def schema_filter(schema: GraphSchema, question: str) -> FilterResult:
        seen.append(question)
        return FilterResult(FILTERED, Usage(30, 6))

    llm = FakeLLM([Completion(reply("a"), Usage(10, 4))])

    result = await run_workflow(
        "Who acted?",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=schema_filter,
    )

    assert seen == ["Who acted?"]
    assert [a.schema_scope for a in result.attempts] == ["filtered"]
    assert llm.calls[0].system == generator_system(FILTERED)
    assert result.filter_usage == Usage(30, 6)
    # the filter's tokens are part of the total, with the attempts'
    assert result.usage == Usage(40, 10)


async def test_an_async_filter_may_return_a_bare_schema():
    async def schema_filter(schema: GraphSchema, question: str) -> GraphSchema:
        return FILTERED

    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([Completion(reply("a"), Usage(1, 1))]),
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=schema_filter,
    )

    assert result.attempts[0].schema_scope == "filtered"
    assert result.filter_usage == Usage(0, 0)
    assert result.usage == Usage(1, 1)


async def test_without_an_llm_filter_nothing_is_added_to_the_usage():
    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([Completion(reply("a"), Usage(10, 4))]),
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.filter_usage == Usage(0, 0)
    assert result.usage == Usage(10, 4)


async def test_the_usage_is_unknown_when_the_filter_did_not_report_its_tokens():
    async def schema_filter(schema: GraphSchema, question: str) -> FilterResult:
        return FilterResult(FILTERED, None)

    result = await run_workflow(
        "Q",
        FULL,
        llm=FakeLLM([Completion(reply("a"), Usage(10, 4))]),
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=schema_filter,
    )

    assert result.filter_usage is None
    assert result.usage is None


async def test_the_filters_tokens_are_kept_when_a_run_stops_in_the_middle():
    from text2cypher import LLMError

    async def schema_filter(schema: GraphSchema, question: str) -> FilterResult:
        return FilterResult(FILTERED, Usage(30, 6))

    llm = FakeLLM([Completion(reply("a"), Usage(10, 4)), ConnectionError("down")])

    with pytest.raises(LLMError) as raised:
        await run_workflow(
            "Q",
            FULL,
            llm=llm,
            validate=ScriptedValidator(invalid("x")),
            config=Config(),
            schema_filter=schema_filter,
        )

    assert raised.value.filter_usage == Usage(30, 6)
    assert raised.value.usage == Usage(40, 10)


READING = Usage(20, 5)


def reads(skill_id: str, usage: Usage | None = READING) -> Completion:
    return Completion("", usage, (ToolCall("c", "read_skill", {"id": skill_id}),))


async def test_the_skills_read_are_on_the_attempt_and_their_tokens_in_its_usage():
    llm = FakeLLM([reads("path-finding"), Completion(reply("a"), Usage(10, 4))])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID),
        config=Config(use_skills=True),
    )

    assert [a.skills for a in result.attempts] == [("path-finding",)]
    assert result.attempts[0].usage == Usage(30, 9)
    assert result.usage == Usage(30, 9)


async def test_a_correction_has_what_was_read_in_its_prompt_and_replays_none_of_the_reading():
    llm = FakeLLM(
        [
            reads("path-finding"),
            Completion(reply("a"), Usage(10, 4)),
            Completion(reply("b"), Usage(7, 2)),
        ]
    )

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(use_skills=True),
    )

    assert result.status == "valid"
    assert [a.skills for a in result.attempts] == [("path-finding",), ()]
    correction = llm.calls[2]
    assert correction.system == generator_system(
        FULL, skills=SkillsOffer(tools=True, read=("path-finding",))
    )
    assert [m.role for m in correction.messages] == ["user", "assistant", "user"]
    assert correction.tools == (READ_SKILL,)


async def test_the_skills_it_read_for_a_failed_query_stay_in_the_prompt_of_every_later_one():
    llm = FakeLLM(
        [
            reads("path-finding"),
            reply("a"),
            reads("functions"),
            reply("b"),
            reply("c"),
        ]
    )

    await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), invalid("y"), VALID),
        config=Config(use_skills=True),
    )

    third = llm.calls[-1].system
    assert "# Path finding" in third
    assert "# Functions" in third
    assert "- graph-algorithms:" in third
    assert "- path-finding:" not in third


async def test_a_run_that_stops_while_reading_skills_keeps_what_the_earlier_calls_cost():
    from text2cypher import LLMError

    llm = FakeLLM([reads("functions"), ConnectionError("down")])

    with pytest.raises(LLMError) as raised:
        await run_workflow(
            "Q",
            FULL,
            llm=llm,
            validate=ScriptedValidator(),
            config=Config(use_skills=True),
        )

    error = raised.value
    assert isinstance(error.__cause__, ConnectionError)
    # a request that made no query, which holds what its calls cost
    assert [(a.query, a.validation, a.usage) for a in error.attempts] == [
        ("", None, Usage(20, 5))
    ]
    assert error.usage == Usage(20, 5)


async def test_without_the_option_nothing_is_offered_to_read():
    llm = FakeLLM([reply("a")])

    result = await run_workflow(
        "Q", FULL, llm=llm, validate=ScriptedValidator(VALID), config=Config()
    )

    assert llm.calls[0].tools == ()
    assert result.attempts[0].skills == ()


async def test_a_reply_with_no_text_is_asked_again_as_it_was_and_costs_no_correction():
    llm = FakeLLM([Completion("", Usage(8, 0)), Completion(reply("a"), Usage(10, 4))])
    validate = ScriptedValidator(VALID)

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=validate,
        config=Config(attempts_w_filtered=0, total_attempts=0),
    )

    assert result.status == "valid"
    assert [(a.query, a.empty_reply) for a in result.attempts] == [
        ("", True),
        ("a", False),
    ]
    assert validate.queries == ["a"]
    assert result.usage == Usage(18, 4)
    # the same request: nothing about it changed
    assert llm.calls[0].system == llm.calls[1].system
    assert llm.calls[0].messages == llm.calls[1].messages


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
async def test_a_reply_of_nothing_but_space_has_no_text_either(text: str):
    llm = FakeLLM([text, text])
    validate = ScriptedValidator()

    result = await run_workflow("Q", FULL, llm=llm, validate=validate, config=Config())

    # twice in a row: the run fails, and it is not a decline
    assert result.status == "failed"
    assert result.query == ""
    assert [a.empty_reply for a in result.attempts] == [True, True]
    assert validate.queries == []
    assert len(llm.calls) == 2


async def test_a_second_reply_with_no_text_in_a_row_ends_the_run_as_failed_with_the_tokens():
    llm = FakeLLM([Completion("", Usage(8, 0)), Completion("", Usage(8, 0))])

    result = await run_workflow(
        "Q", FULL, llm=llm, validate=ScriptedValidator(), config=Config()
    )

    assert result.status == "failed"
    assert result.usage == Usage(16, 0)
    assert all(a.validation is None for a in result.attempts)


async def test_an_explicit_empty_block_is_still_a_decline_and_not_an_empty_reply():
    llm = FakeLLM(["```cypher\n```"])

    result = await run_workflow(
        "Delete all", FULL, llm=llm, validate=ScriptedValidator(), config=Config()
    )

    assert result.status == "unanswerable"
    assert [a.empty_reply for a in result.attempts] == [False]


async def test_a_reply_with_no_text_then_a_decline_on_the_filtered_schema_goes_on_to_the_full_one():
    llm = FakeLLM(["", "```cypher\n```", reply("a")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(VALID),
        config=Config(),
        schema_filter=lambda schema, question: FILTERED,
    )

    assert result.status == "valid"
    assert [(a.query, a.schema_scope, a.empty_reply) for a in result.attempts] == [
        ("", "filtered", True),
        ("", "filtered", False),
        ("a", "full", False),
    ]
    assert [c.system for c in llm.calls] == [
        generator_system(FILTERED),
        generator_system(FILTERED),
        generator_system(FULL),
    ]


async def test_an_empty_reply_to_a_correction_is_asked_again_with_the_same_turns():
    llm = FakeLLM([reply("a"), "", reply("b")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )

    assert result.status == "valid"
    assert llm.calls[1].messages == llm.calls[2].messages
    assert len(llm.calls[1].messages) == 3


async def test_one_empty_reply_is_forgiven_each_time_and_not_once_for_the_whole_run():
    llm = FakeLLM(["", reply("a"), "", reply("b")])

    result = await run_workflow(
        "Q",
        FULL,
        llm=llm,
        validate=ScriptedValidator(invalid("x"), VALID),
        config=Config(),
    )

    assert result.status == "valid"
    assert [a.empty_reply for a in result.attempts] == [True, False, True, False]
