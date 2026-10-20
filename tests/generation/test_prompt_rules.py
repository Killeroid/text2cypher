"""The FalkorDB rules in the generator prompt match what the validator and FalkorDB do.

Each rule the prompt states is checked here on the embedded server, so a change in
cypher-guardrail or FalkorDB that makes a rule wrong fails a test.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from cypher_guardrail import issues as cg
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from text2cypher._generation.prompts import (
    EXAMPLE_QUERY,
    FULLTEXT_EXAMPLE,
    PARAMETER_EXAMPLE,
    SHORTEST_PATH_EXAMPLE,
)
from text2cypher._validation import QueryValidator

pytestmark = pytest.mark.anyio


@pytest.fixture
async def cities(falkordb_client: FalkorDB) -> AsyncIterator[str]:
    name = f"cities_{uuid.uuid4().hex[:8]}"
    graph = falkordb_client.select_graph(name)
    await graph.query(
        "CREATE (:City {name: 'Paris'})-[:ROAD]->(:City {name: 'Lyon'})"
        "-[:ROAD]->(:City {name: 'Berlin'})"
    )
    yield name
    await graph.delete()


async def codes(falkordb_client: FalkorDB, graph: str, query: str) -> list[str]:
    result = await QueryValidator(falkordb_client).validate(query, graph)
    return [issue.code for issue in result.issues]


async def test_a_query_that_writes_is_not_valid(falkordb_client, cities):
    assert await codes(falkordb_client, cities, "CREATE (:City {name: 'Nice'})") == [
        cg.WRITE_QUERY_ERROR
    ]


async def test_a_parameter_without_a_value_is_not_valid(falkordb_client, cities):
    query = "MATCH (c:City {name: $name}) RETURN c.name"

    assert await codes(falkordb_client, cities, query) == [cg.PARAMETER_NOT_PROVIDED]


async def test_the_shortest_path_example_is_valid_and_runs(falkordb_client, cities):
    assert await codes(falkordb_client, cities, SHORTEST_PATH_EXAMPLE) == []

    result = await falkordb_client.select_graph(cities).ro_query(SHORTEST_PATH_EXAMPLE)

    assert result.result_set == [[["Paris", "Lyon", "Berlin"]]]


async def test_shortest_path_in_match_is_not_valid(falkordb_client, cities):
    query = (
        "MATCH (a:City {name: 'Paris'}), (b:City {name: 'Berlin'}) "
        "MATCH p = shortestPath((a)-[:ROAD*]->(b)) RETURN [n IN nodes(p) | n.name]"
    )

    assert await codes(falkordb_client, cities, query) == [cg.EXECUTION_FAILED]


@pytest.mark.parametrize("query", ["", "   \n"])
async def test_an_empty_query_is_not_valid(falkordb_client, cities, query):
    assert await codes(falkordb_client, cities, query) == [cg.SYNTAX_ERROR]


async def test_the_example_query_is_valid_on_the_example_schema(falkordb_client):
    name = f"example_{uuid.uuid4().hex[:8]}"
    graph = falkordb_client.select_graph(name)
    await graph.query(
        "CREATE (:Employee {name: 'Ada'})-[:WORKS_IN]->(:Department {name: 'Sales'})"
    )
    try:
        assert await codes(falkordb_client, name, EXAMPLE_QUERY) == []
        assert (await graph.ro_query(EXAMPLE_QUERY)).result_set == [["Ada"]]
    finally:
        await graph.delete()


@pytest.fixture
async def indexed(falkordb_client: FalkorDB) -> AsyncIterator[AsyncGraph]:
    """People with range indexes, movies with a full-text index, documents with a vector index."""
    graph = falkordb_client.select_graph(f"indexed_{uuid.uuid4().hex[:8]}")
    await graph.query(
        "UNWIND range(1, 50) AS i CREATE (:Person {name: 'P' + toString(i), born: 1900 + i})"
    )
    await graph.query(
        "CREATE (:Movie {title: 'The Matrix', plot: 'a hacker learns the truth'}), "
        "(:Doc {text: 'x', embedding: vecf32([1.0, 2.0, 3.0, 4.0])})"
    )
    await graph.query("CREATE INDEX FOR (p:Person) ON (p.name)")
    await graph.query("CREATE INDEX FOR (p:Person) ON (p.born)")
    await graph.query("CALL db.idx.fulltext.createNodeIndex('Movie', 'title', 'plot')")
    await graph.query(
        "CREATE VECTOR INDEX FOR (d:Doc) ON (d.embedding) "
        "OPTIONS {dimension: 4, similarityFunction: 'euclidean'}"
    )
    yield graph
    await graph.delete()


async def uses_an_index(graph: AsyncGraph, query: str) -> bool:
    return any("Index Scan" in step for step in await graph.explain(query))


@pytest.mark.parametrize(
    "where",
    [
        "p.name = 'P3'",
        "p.born > 1940",
        "p.born >= 1920 AND p.born < 1940",
        "p.name IN ['P1', 'P2']",
        "p.name = 'P1' OR p.name = 'P2'",
    ],
)
async def test_a_direct_comparison_on_an_indexed_property_uses_the_index(
    indexed: AsyncGraph, where: str
):
    assert await uses_an_index(indexed, f"MATCH (p:Person) WHERE {where} RETURN p.born")


@pytest.mark.parametrize(
    "where",
    [
        "p.name <> 'P3'",
        "NOT p.name = 'P3'",
        "toLower(p.name) = 'p3'",
        "p.name STARTS WITH 'P1'",
        "p.name CONTAINS '1'",
        "p.born + 1 > 1941",
    ],
)
async def test_these_ways_of_asking_do_not_use_the_index(
    indexed: AsyncGraph, where: str
):
    assert not await uses_an_index(
        indexed, f"MATCH (p:Person) WHERE {where} RETURN p.born"
    )


async def test_the_inline_form_of_an_equality_uses_the_index_too(indexed: AsyncGraph):
    assert await uses_an_index(indexed, "MATCH (p:Person {name: 'P3'}) RETURN p.born")


async def test_not_equal_is_written_with_the_angle_brackets(
    falkordb_client, indexed: AsyncGraph
):
    right = "MATCH (p:Person) WHERE p.name <> 'P3' RETURN p.born"
    wrong = "MATCH (p:Person) WHERE p.name != 'P3' RETURN p.born"

    assert await codes(falkordb_client, indexed.name, right) == []
    assert await codes(falkordb_client, indexed.name, wrong) == [cg.SYNTAX_ERROR]


async def test_the_fulltext_example_is_valid_and_finds_the_word(
    falkordb_client, indexed: AsyncGraph
):
    assert await codes(falkordb_client, indexed.name, FULLTEXT_EXAMPLE) == []

    result = await indexed.ro_query(FULLTEXT_EXAMPLE)

    assert result.result_set == [["The Matrix"]]


async def test_a_vector_query_is_valid_so_only_the_prompt_keeps_the_model_from_it(
    falkordb_client, indexed: AsyncGraph
):
    query = (
        "CALL db.idx.vector.queryNodes('Doc', 'embedding', 1, "
        "vecf32([1.0, 2.0, 3.0, 4.0])) YIELD node RETURN node.text"
    )

    assert await codes(falkordb_client, indexed.name, query) == []


@pytest.fixture
async def with_a_function(falkordb_client: FalkorDB, cities: str) -> AsyncIterator[str]:
    """The cities graph, on an instance that has one user-defined function."""
    await falkordb_client.connection.execute_command(
        "GRAPH.UDF",
        "LOAD",
        "REPLACE",
        "rules_lib",
        "function double(x) { return x * 2; } falkor.register('double', double);",
    )
    yield cities
    await falkordb_client.connection.execute_command("GRAPH.UDF", "DELETE", "rules_lib")


async def test_a_listed_function_is_valid_and_runs_and_another_is_not(
    falkordb_client, with_a_function
):
    graph = falkordb_client.select_graph(with_a_function)
    listed = "MATCH (c:City {name: 'Paris'}) RETURN rules_lib.double(2)"
    unlisted = "MATCH (c:City {name: 'Paris'}) RETURN rules_lib.triple(2)"

    assert await codes(falkordb_client, with_a_function, listed) == []
    assert (await graph.ro_query(listed)).result_set == [[4]]
    assert set(await codes(falkordb_client, with_a_function, unlisted)) == {
        cg.UNKNOWN_FUNCTION_ERROR,
        cg.UNKNOWN_FUNCTION,
    }


async def test_the_parameter_example_is_valid_and_runs(falkordb_client, cities):
    graph = falkordb_client.select_graph(cities)
    await graph.query("CREATE (:Person {name: 'Ann', born: 1990})")
    query = PARAMETER_EXAMPLE

    assert await codes(falkordb_client, cities, query) == []
    assert (await graph.ro_query(query)).result_set == [[1990]]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "CYPHER name='Paris', year=1999 MATCH (c:City {name: $name}) RETURN c",
            [cg.SYNTAX_ERROR],
        ),
        (
            "CYPHER name='Paris' MATCH (c:City {name: $other}) RETURN c",
            [cg.PARAMETER_NOT_PROVIDED],
        ),
        ('CYPHER title="You\'ve" MATCH (c:City {name: $title}) RETURN c', []),
        (
            "CYPHER names=['Paris'] MATCH (c:City) WHERE c.name IN $names RETURN c.name",
            [],
        ),
        ("CYPHER n=1 MATCH (c:City) RETURN c.name LIMIT $n", []),
    ],
    ids=[
        "commas",
        "a parameter that is not in the header",
        "an apostrophe",
        "a list",
        "limit",
    ],
)
async def test_what_the_parameter_rule_says_about_the_header(
    falkordb_client, cities, query, expected
):
    got = await codes(falkordb_client, cities, query)

    assert [code.rsplit(".", 1)[-1] for code in got] == [
        code.rsplit(".", 1)[-1] for code in expected
    ]
