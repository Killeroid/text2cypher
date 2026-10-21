from collections import Counter

import pytest
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import ConnectionError as RedisConnectionError
from support import reply

from text2cypher import Config, FakeLLM, GraphNotFoundError, Text2Cypher
from text2cypher._results import Records

pytestmark = pytest.mark.anyio

# Schema of the ``movies_graph`` fixture:
#   (:Person {name})-[:ACTED_IN {role}]->(:Movie {title, year})


async def test_a_valid_query_comes_back_on_the_first_try(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name")])

    result = await Text2Cypher(falkordb_client, llm).run("Who acted?", movies_graph)

    assert result.status == "valid"
    assert len(result.attempts) == 1
    assert result.attempts[0].validation is not None
    assert result.attempts[0].validation.valid
    # the prompt carries the real schema of the graph, found through a snapshot
    assert (
        "Movie {title: STRING (e.g. 'Matrix'), year: INTEGER (e.g. 1999)}"
        in llm.calls[0].system
    )
    assert "(:Person)-[:ACTED_IN]->(:Movie)" in llm.calls[0].system


async def test_a_wrong_first_query_is_corrected_with_the_validators_findings(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN p.name"),
            reply("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name"),
        ]
    )

    result = await Text2Cypher(falkordb_client, llm).run("Who acted?", movies_graph)

    assert result.status == "valid"
    assert [a.query for a in result.attempts] == [
        "MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN p.name",
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name",
    ]
    # what the validator found about the first query reached the corrector
    assert "(m:Movie)<-[:ACTED_IN]-(p:Person)" in llm.calls[1].user


async def test_a_query_that_writes_is_not_accepted(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("CREATE (:Person {name: 'x'})")] * 2)

    result = await Text2Cypher(
        falkordb_client, llm, config=Config(attempts_w_filtered=0, total_attempts=1)
    ).run("Add a person", movies_graph)

    assert result.status == "failed"
    assert len(result.attempts) == 2


async def test_a_request_the_llm_declines_is_unanswerable(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(["```cypher\n```"])

    result = await Text2Cypher(falkordb_client, llm).run(
        "Delete everything", movies_graph
    )

    assert result.status == "unanswerable"
    assert len(llm.calls) == 1


async def test_the_prompt_has_only_the_part_of_the_schema_the_question_is_about(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (:Movie)-[:IN_GENRE]->(g:Genre) RETURN g.name")])

    result = await Text2Cypher(falkordb_client, llm).run(
        "Which genre is the Matrix in?", film_graph.name
    )

    assert result.status == "valid"
    assert result.attempts[0].schema_scope == "filtered"
    system = llm.calls[0].system
    assert "Genre {name: STRING (e.g. 'SciFi')}" in system
    assert "Movie {" in system
    assert "Person" not in system
    assert "KNOWS" not in system


async def test_without_a_filter_the_prompt_has_the_whole_schema(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (:Movie)-[:IN_GENRE]->(g:Genre) RETURN g.name")])

    result = await Text2Cypher(falkordb_client, llm, schema_filter=None).run(
        "Which genre is the Matrix in?", film_graph.name
    )

    assert result.attempts[0].schema_scope == "full"
    assert "Person" in llm.calls[0].system
    assert "KNOWS" in llm.calls[0].system


async def test_a_decline_on_the_filtered_schema_is_asked_again_on_the_whole_one(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    # the filter keeps Genre and Movie, which cannot tell who is related to whom
    llm = FakeLLM(
        [
            "```cypher\n```",
            reply("MATCH (:Person)-[:KNOWS]->(p:Person) RETURN p.name"),
        ]
    )

    result = await Text2Cypher(falkordb_client, llm).run(
        "Which genre does Keanu's friend like?", film_graph.name
    )

    assert result.status == "valid"
    assert [a.schema_scope for a in result.attempts] == ["filtered", "full"]
    assert "KNOWS" not in llm.calls[0].system
    assert "KNOWS" in llm.calls[1].system


async def test_by_default_nothing_is_executed(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    result = await Text2Cypher(falkordb_client, llm).run("Who?", movies_graph)

    assert result.status == "valid"
    assert result.records is None
    assert result.execution_error is None


async def test_a_valid_query_is_executed_when_asked_to(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "Who?", movies_graph
    )

    assert result.status == "valid"
    assert result.records == Records(("p.name",), (("Keanu",),))
    assert result.execution_error is None


@pytest.mark.parametrize(
    ("replies", "status"),
    [(["```cypher\n```"], "unanswerable"), ([reply("CREATE (:X)")] * 2, "failed")],
)
async def test_a_query_that_is_not_valid_is_not_executed(
    falkordb_client: FalkorDB, movies_graph: str, replies: list[str], status: str
):
    result = await Text2Cypher(
        falkordb_client,
        FakeLLM(replies),
        config=Config(execute=True, attempts_w_filtered=0, total_attempts=1),
    ).run("Q", movies_graph)

    assert result.status == status
    assert result.records is None
    assert result.execution_error is None


async def test_a_query_that_fails_when_executed_keeps_its_status_and_has_the_error(
    falkordb_client: FalkorDB, movies_graph: str
):
    slow = reply("UNWIND range(1, 50000000) AS x RETURN count(x)")

    result = await Text2Cypher(
        falkordb_client,
        FakeLLM([slow]),
        config=Config(execute=True, execute_timeout_ms=1),
    ).run("How many?", movies_graph)

    assert result.status == "valid"
    assert result.records is None
    assert result.execution_error is not None
    assert "timed out" in result.execution_error


async def test_a_lost_connection_during_execution_does_not_lose_the_result(
    falkordb_client: FalkorDB, movies_graph: str, monkeypatch: pytest.MonkeyPatch
):
    async def connection_lost(*args: object, **kwargs: object):
        raise RedisConnectionError("connection lost")

    monkeypatch.setattr("text2cypher._execution.execute_cypher", connection_lost)
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "Who?", movies_graph
    )

    # the LLM calls were paid for: the valid query and its attempt are still there
    assert result.status == "valid"
    assert result.query == "MATCH (p:Person) RETURN p.name"
    assert len(result.attempts) == 1
    assert result.records is None
    assert result.execution_error == "connection lost"


async def test_a_programming_error_during_execution_is_not_swallowed(
    falkordb_client: FalkorDB, movies_graph: str, monkeypatch: pytest.MonkeyPatch
):
    async def broken(*args: object, **kwargs: object):
        raise TypeError("a bug")

    monkeypatch.setattr("text2cypher._execution.execute_cypher", broken)
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    with pytest.raises(TypeError, match="a bug"):
        await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
            "Who?", movies_graph
        )


async def test_a_graph_that_does_not_exist_is_an_error_before_any_llm_call(
    falkordb_client: FalkorDB,
):
    llm = FakeLLM([reply("MATCH (n) RETURN n")])

    with pytest.raises(GraphNotFoundError, match="no_such_graph"):
        await Text2Cypher(falkordb_client, llm).run("Who?", "no_such_graph")

    assert llm.calls == []


async def test_the_schema_is_read_again_on_every_run_by_default(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (n) RETURN n")] * 2)
    agent = Text2Cypher(falkordb_client, llm)
    await agent.run("Who?", movies_graph)
    await falkordb_client.select_graph(movies_graph).query(
        "CREATE (:Studio {name: 's'})"
    )

    await agent.run("Who?", movies_graph)

    assert "Studio" in llm.calls[1].system


async def test_a_cached_schema_is_not_read_again_before_it_is_due(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (n) RETURN n")] * 2)
    agent = Text2Cypher(
        falkordb_client, llm, config=Config(schema_refresh_seconds=3600)
    )
    await agent.run("Who?", movies_graph)
    await falkordb_client.select_graph(movies_graph).query(
        "CREATE (:Studio {name: 's'})"
    )

    await agent.run("Who?", movies_graph)

    assert "Studio" not in llm.calls[1].system


async def test_a_schema_that_is_due_is_checked_and_read_again_if_the_graph_changed(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (n) RETURN n")] * 3)
    agent = Text2Cypher(falkordb_client, llm, config=Config(schema_refresh_seconds=0))
    await agent.run("Who?", movies_graph)
    await agent.run("Who?", movies_graph)  # nothing changed
    assert "Studio" not in llm.calls[1].system
    await falkordb_client.select_graph(movies_graph).query(
        "CREATE (:Studio {name: 's'})"
    )

    await agent.run("Who?", movies_graph)

    assert "Studio" in llm.calls[2].system


async def test_a_cached_graph_that_never_existed_is_still_an_error(
    falkordb_client: FalkorDB,
):
    llm = FakeLLM([reply("MATCH (n) RETURN n")])
    agent = Text2Cypher(falkordb_client, llm, config=Config(schema_refresh_seconds=60))

    for _ in range(2):
        with pytest.raises(GraphNotFoundError, match="no_such_graph"):
            await agent.run("Who?", "no_such_graph")

    assert llm.calls == []


async def test_a_run_on_a_cached_schema_only_asks_the_database_to_plan_the_query(
    falkordb_client: FalkorDB, movies_graph: str, commands: Counter[str]
):
    llm = FakeLLM([reply("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name")] * 2)
    agent = Text2Cypher(
        falkordb_client, llm, config=Config(schema_refresh_seconds=3600)
    )
    first = await agent.run("Who acted?", movies_graph)
    commands.clear()

    second = await agent.run("Who acted?", movies_graph)

    assert first.status == second.status == "valid"
    assert sum(commands.values()) == 1


async def test_a_graph_created_after_a_missing_one_was_reported_is_found(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (n:Person) RETURN n")])
    agent = Text2Cypher(
        falkordb_client, llm, config=Config(schema_refresh_seconds=3600)
    )
    with pytest.raises(GraphNotFoundError):
        await agent.run("Who?", new_graph.name)
    await new_graph.query("CREATE (:Person {name: 'Keanu'})")

    result = await agent.run("Who?", new_graph.name)

    assert result.status == "valid"


async def test_the_indexes_of_the_graph_are_in_the_prompt(
    falkordb_client: FalkorDB, movies_graph: str
):
    await falkordb_client.select_graph(movies_graph).query(
        "CREATE INDEX FOR (p:Person) ON (p.name)"
    )
    llm = FakeLLM([reply("MATCH (p:Person {name: 'Keanu'}) RETURN p.name")])

    result = await Text2Cypher(falkordb_client, llm).run("Who is Keanu?", movies_graph)

    assert result.status == "valid"
    assert "Indexes:\nPerson.name: RANGE" in llm.calls[0].system
    assert "cannot use the index" in llm.calls[0].system


async def test_the_performance_check_makes_the_loop_correct_a_cartesian_product(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (p:Person), (m:Movie) RETURN p.name, m.title"),
            reply("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title"),
        ]
    )

    result = await Text2Cypher(
        falkordb_client, llm, config=Config(check_performance=True)
    ).run("Who acted in what?", movies_graph)

    assert result.status == "valid"
    assert len(result.attempts) == 2
    assert "are not connected" in llm.calls[1].user


async def test_a_sum_of_text_is_corrected_without_asking_for_the_type_check(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (m:Movie) RETURN sum(m.title)"),
            reply("MATCH (m:Movie) RETURN sum(m.year)"),
        ]
    )

    result = await Text2Cypher(falkordb_client, llm).run(
        "Total of the years?", movies_graph
    )

    assert result.status == "valid"
    assert len(result.attempts) == 2


async def test_the_type_check_can_be_turned_off(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (m:Movie) RETURN sum(m.title)")])

    result = await Text2Cypher(
        falkordb_client, llm, config=Config(check_types=False)
    ).run("Total of the titles?", movies_graph)

    assert result.status == "valid"
    assert len(result.attempts) == 1


async def test_the_type_check_makes_the_loop_correct_a_sum_of_text(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (m:Movie) RETURN sum(m.title)"),
            reply("MATCH (m:Movie) RETURN sum(m.year)"),
        ]
    )

    result = await Text2Cypher(
        falkordb_client, llm, config=Config(check_types=True)
    ).run("What is the total of the years?", movies_graph)

    assert result.status == "valid"
    assert "`sum` needs numbers" in llm.calls[1].user


async def test_the_user_defined_functions_of_the_instance_are_in_the_prompt(
    falkordb_client: FalkorDB, movies_graph: str
):
    await falkordb_client.connection.execute_command(
        "GRAPH.UDF",
        "LOAD",
        "REPLACE",
        "pipeline_lib",
        "function double(x) { return x * 2; } falkor.register('double', double);",
    )
    try:
        llm = FakeLLM([reply("MATCH (m:Movie) RETURN pipeline_lib.double(m.year)")])

        result = await Text2Cypher(falkordb_client, llm).run(
            "Double the years", movies_graph
        )
    finally:
        await falkordb_client.connection.execute_command(
            "GRAPH.UDF", "DELETE", "pipeline_lib"
        )

    assert result.status == "valid"
    assert "User-defined functions:\npipeline_lib.double" in llm.calls[0].system


async def test_a_query_with_a_parameter_header_is_validated_and_executed(
    falkordb_client: FalkorDB, movies_graph: str
):
    query = "CYPHER name='Keanu' MATCH (p:Person {name: $name}) RETURN p.name"
    llm = FakeLLM([reply(query)])

    result = await Text2Cypher(
        falkordb_client, llm, config=Config(use_parameters=True, execute=True)
    ).run("Who is Keanu?", movies_graph)

    assert result.status == "valid"
    assert result.query == query
    assert result.records is not None
    assert result.records.rows == (("Keanu",),)
    assert "CYPHER header" in llm.calls[0].system


async def test_a_query_that_fails_on_the_database_is_corrected_with_the_error(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (m:Movie) RETURN m.year % 0"),
            reply("MATCH (m:Movie) RETURN m.year % 100"),
        ]
    )

    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "What is the year modulo 100?", movies_graph
    )

    assert result.status == "valid"
    assert result.query == "MATCH (m:Movie) RETURN m.year % 100"
    assert result.records is not None
    assert result.records.rows == ((99,),)
    assert [a.execution_error for a in result.attempts] == ["Division by zero", None]
    # the model is told what the database said, after the query it wrote
    assert [m.role for m in llm.calls[1].messages] == ["user", "assistant", "user"]
    assert "failed when it ran:\n\nDivision by zero" in llm.calls[1].user


async def test_without_execution_a_query_that_would_fail_on_the_database_is_valid(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (m:Movie) RETURN m.year % 0")])

    result = await Text2Cypher(falkordb_client, llm).run("Q", movies_graph)

    assert result.status == "valid"
    assert len(llm.calls) == 1


async def test_a_query_that_keeps_failing_on_the_database_ends_as_failed(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (m:Movie) RETURN m.year % 0")] * 2)

    result = await Text2Cypher(
        falkordb_client,
        llm,
        config=Config(execute=True, attempts_w_filtered=1, total_attempts=1),
    ).run("Q", movies_graph)

    assert result.status == "failed"
    assert result.records is None
    assert result.execution_error == "Division by zero"
    assert len(result.attempts) == 2


async def test_the_prompt_shows_example_values_and_how_to_use_them_by_default(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    await Text2Cypher(falkordb_client, llm).run("Who?", film_graph.name)

    system = llm.calls[0].system
    assert "name: STRING (e.g. 'Keanu', 'Carrie')" in system
    assert "ACTED_IN {role: STRING (e.g. 'Neo', 'Trinity')" in system
    assert "write a value the way the examples are written" in system


async def test_with_zero_example_values_nothing_from_the_graph_is_in_the_prompt(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")])

    await Text2Cypher(falkordb_client, llm, config=Config(example_values=0)).run(
        "Who?", film_graph.name
    )

    system = llm.calls[0].system
    for value in ("Keanu", "Carrie", "Neo", "Trinity", "Matrix", "SciFi", "e.g."):
        assert value not in system
    assert "the way the examples are written" not in system


async def test_a_typo_is_fixed_on_a_real_graph_without_asking_the_llm_again(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM([reply("MATCH (p:Persn)-[:ACTD_IN]->(m:Movi) RETURN p.nme, m.titel")])

    result = await Text2Cypher(
        falkordb_client,
        llm,
        config=Config(autofix_typos=True, execute=True, example_values=0),
    ).run("Who acted in what?", movies_graph)

    assert result.status == "valid"
    assert result.query == (
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title"
    )
    assert len(llm.calls) == 1
    assert [a.auto_fixed for a in result.attempts] == [False, True]
    assert result.records == Records(("p.name", "m.title"), (("Keanu", "Matrix"),))


async def test_without_the_option_the_same_typos_are_corrected_by_the_llm(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (p:Persn) RETURN p.name"),
            reply("MATCH (p:Person) RETURN p.name"),
        ]
    )

    result = await Text2Cypher(falkordb_client, llm).run("Who?", movies_graph)

    assert result.status == "valid"
    assert len(llm.calls) == 2
    assert not any(a.auto_fixed for a in result.attempts)


async def test_a_parameter_header_with_commas_is_repaired_on_a_real_graph(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply(
                "name='Keanu', n=1 MATCH (p:Person {name: $name}) RETURN p.name, $n AS n"
            )
        ]
    )

    result = await Text2Cypher(
        falkordb_client,
        llm,
        config=Config(use_parameters=True, execute=True, example_values=0),
    ).run("Who is Keanu?", movies_graph)

    assert result.status == "valid"
    assert len(llm.calls) == 1
    assert result.query == (
        "CYPHER name='Keanu' n=1 MATCH (p:Person {name: $name}) RETURN p.name, $n AS n"
    )
    assert [a.auto_fixed for a in result.attempts] == [False, True]
    assert result.records == Records(("p.name", "n"), (("Keanu", 1),))


async def test_clearing_the_schema_makes_the_next_run_read_the_changed_graph(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (m:Movie) RETURN m.title")] * 3)
    agent = Text2Cypher(
        falkordb_client,
        llm,
        config=Config(schema_refresh_seconds=3600, example_values=0),
        schema_filter=None,
    )
    await agent.run("Which movies?", film_graph.name)
    await film_graph.query("CREATE (:Award {name: 'Oscar'})")

    await agent.run("Which movies?", film_graph.name)
    agent.clear_schema(film_graph.name)
    await agent.run("Which movies?", film_graph.name)

    # the schema is kept, so the new label is not seen until it is cleared
    assert "Award" not in llm.calls[1].system
    assert "Award {name: STRING}" in llm.calls[2].system


async def test_clearing_the_schema_reads_the_example_values_again(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    llm = FakeLLM([reply("MATCH (g:Genre) RETURN g.name")] * 3)
    agent = Text2Cypher(falkordb_client, llm)
    await agent.run("Which genres?", film_graph.name)
    await film_graph.query("MATCH (g:Genre) SET g.name = 'Thriller'")

    await agent.run("Which genres?", film_graph.name)
    agent.clear_schema()
    await agent.run("Which genres?", film_graph.name)

    # the same structure keeps its examples, and clearing reads them again
    assert "Genre {name: STRING (e.g. 'SciFi'" in llm.calls[1].system
    assert "Genre {name: STRING (e.g. 'Thriller'" in llm.calls[2].system


async def test_clearing_one_graph_leaves_the_others_kept(
    falkordb_client: FalkorDB, film_graph: AsyncGraph, movies_graph: str, commands
):
    llm = FakeLLM([reply("MATCH (p:Person) RETURN p.name")] * 4)
    agent = Text2Cypher(
        falkordb_client, llm, config=Config(schema_refresh_seconds=3600)
    )
    await agent.run("Who?", film_graph.name)
    await agent.run("Who?", movies_graph)

    commands.clear()
    agent.clear_schema(film_graph.name)
    await agent.run("Who?", movies_graph)
    kept = sum(commands.values())
    commands.clear()
    await agent.run("Who?", film_graph.name)

    # the graph that was kept costs the one request to plan the query; the cleared one is read again
    assert kept == 1
    assert sum(commands.values()) > kept


async def test_clearing_a_graph_that_was_never_read_is_fine(
    falkordb_client: FalkorDB,
):
    agent = Text2Cypher(falkordb_client, FakeLLM([]))

    agent.clear_schema("never_read")
    agent.clear_schema()


async def test_the_troubleshooting_page_reads_a_result_with_the_names_it_documents(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            reply("MATCH (p:Persn) RETURN p.name"),
            reply("MATCH (p:Person) RETURN p.name"),
        ]
    )
    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "Who?", movies_graph
    )

    lines = []
    for number, attempt in enumerate(result.attempts, start=1):
        lines.append(
            f"{number}. [{attempt.schema_scope}] {attempt.query or '(declined)'}"
        )
        if attempt.auto_fixed:
            lines.append("   repaired without the LLM")
        if attempt.execution_error:
            lines.append(f"   FalkorDB: {attempt.execution_error}")
        if attempt.validation:
            lines.extend(
                f"   {issue.code}: {issue.description}"
                for issue in attempt.validation.issues
            )

    assert lines[0] == "1. [full] MATCH (p:Persn) RETURN p.name"
    assert any("UnknownLabel" in line for line in lines)
    assert lines[-1] == "2. [full] MATCH (p:Person) RETURN p.name"
