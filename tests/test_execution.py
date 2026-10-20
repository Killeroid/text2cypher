import pytest
from falkordb.asyncio import FalkorDB
from redis.exceptions import ResponseError

from text2cypher._execution import execute_cypher
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
