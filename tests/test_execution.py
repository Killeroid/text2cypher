import pytest
from falkordb.asyncio import FalkorDB
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from text2cypher._execution import Execution, execute_cypher, run_query
from text2cypher._results import Records

pytestmark = pytest.mark.anyio

# Schema of the ``movies_graph`` fixture:
#   (:Person {name})-[:ACTED_IN {role}]->(:Movie {title, year})


async def test_it_returns_the_columns_and_the_rows(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    records = await execute_cypher(
        graph,
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title AS title, count(m)",
    )

    assert records == Records(
        ("p.name", "title", "count(m)"), (("Keanu", "Matrix", 1),)
    )


async def test_a_query_with_no_rows_still_has_its_columns(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    records = await execute_cypher(
        graph, "MATCH (p:Person {name: 'Nobody'}) RETURN p.name"
    )

    assert records == Records(("p.name",), ())


async def test_it_cannot_write(falkordb_client: FalkorDB, movies_graph: str):
    graph = falkordb_client.select_graph(movies_graph)

    with pytest.raises(ResponseError, match="read-only"):
        await execute_cypher(graph, "CREATE (:Person {name: 'Intruder'})")

    records = await execute_cypher(graph, "MATCH (p:Person) RETURN count(p)")
    assert records.rows == ((1,),)


async def test_it_gives_up_after_the_timeout(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    with pytest.raises(ResponseError, match="timed out"):
        await execute_cypher(
            graph,
            "UNWIND range(1, 50000000) AS x RETURN count(x)",
            timeout_ms=1,
        )


async def test_run_query_gives_the_rows_when_it_runs(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    execution = await run_query(graph, "MATCH (p:Person) RETURN p.name")

    assert execution == Execution(records=Records(("p.name",), (("Keanu",),)))


async def test_run_query_gives_the_error_of_a_query_that_fails_and_says_it_can_be_corrected(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    execution = await run_query(graph, "MATCH (m:Movie) RETURN m.year % 0")

    assert execution == Execution(error="Division by zero", correctable=True)


async def test_run_query_says_a_timeout_cannot_be_corrected(
    falkordb_client: FalkorDB, movies_graph: str
):
    graph = falkordb_client.select_graph(movies_graph)

    execution = await run_query(
        graph, "UNWIND range(1, 50000000) AS x RETURN count(x)", timeout_ms=1
    )

    assert execution.records is None
    assert execution.error is not None
    assert "timed out" in execution.error
    assert not execution.correctable


async def test_run_query_says_a_lost_connection_cannot_be_corrected(
    falkordb_client: FalkorDB, movies_graph: str, monkeypatch: pytest.MonkeyPatch
):
    async def connection_lost(*args: object, **kwargs: object):
        raise RedisConnectionError("connection lost")

    monkeypatch.setattr("text2cypher._execution.execute_cypher", connection_lost)

    execution = await run_query(falkordb_client.select_graph(movies_graph), "RETURN 1")

    assert execution == Execution(error="connection lost", correctable=False)
