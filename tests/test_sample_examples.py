from unittest.mock import AsyncMock

import pytest
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import ResponseError

from text2cypher._examples import _first_values, _literal, sample_examples
from text2cypher._schema_source import SchemaSource
from text2cypher.schema import GraphSchema

pytestmark = pytest.mark.anyio


def schema_of(**properties: tuple[str, ...]) -> GraphSchema:
    return GraphSchema({"Item": dict(properties)}, {}, ())


async def test_it_gives_the_distinct_values_of_a_property_up_to_the_count(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    await new_graph.query(
        "UNWIND ['a', 'b', 'a', 'c', 'd', 'b'] AS v CREATE (:Item {name: v})"
    )

    nodes, relationships = await sample_examples(
        falkordb_client, new_graph.name, schema_of(name=("String",)), 3
    )

    assert nodes == {"Item": {"name": ("'a'", "'b'", "'c'")}}
    assert relationships == {}


async def test_it_writes_each_value_the_way_a_query_does(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    await new_graph.query(
        'CREATE (:Item {text: "it\'s", n: 7, f: 1.5, b: true, '
        "long: '" + "x" * 60 + "'})"
    )

    nodes, _ = await sample_examples(
        falkordb_client,
        new_graph.name,
        schema_of(
            text=("String",),
            n=("Integer",),
            f=("Float",),
            b=("Boolean",),
            long=("String",),
        ),
        3,
    )

    assert nodes["Item"] == {
        "text": ("'it\\'s'",),
        "n": ("7",),
        "f": ("1.5",),
        "b": ("true",),
        "long": (f"'{'x' * 40}…'",),
    }


async def test_it_gives_the_examples_of_a_relationship_property_too(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    await new_graph.query(
        "CREATE (:A)-[:KNOWS {since: 1999}]->(:B), (:A)-[:KNOWS {since: 2001}]->(:B)"
    )
    schema = GraphSchema({}, {"KNOWS": {"since": ("Integer",)}}, ())

    nodes, relationships = await sample_examples(
        falkordb_client, new_graph.name, schema, 5
    )

    assert nodes == {}
    assert relationships == {"KNOWS": {"since": ("1999", "2001")}}


async def test_a_property_that_is_not_plain_text_has_no_examples(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    await new_graph.query("CREATE (:Item {tags: ['a', 'b'], name: 'x'})")

    nodes, _ = await sample_examples(
        falkordb_client,
        new_graph.name,
        schema_of(tags=("List",), name=("String",), missing=("String",)),
        3,
    )

    # a list is not sampled, and a property that no node has gives nothing
    assert nodes == {"Item": {"name": ("'x'",)}}


async def test_a_name_that_needs_quotes_is_sampled(
    falkordb_client: FalkorDB, new_graph: AsyncGraph
):
    await new_graph.query("CREATE (:`My Label` {`a b`: 'v'})")
    schema = GraphSchema({"My Label": {"a b": ("String",)}}, {}, ())

    nodes, _ = await sample_examples(falkordb_client, new_graph.name, schema, 3)

    assert nodes == {"My Label": {"a b": ("'v'",)}}


async def test_a_query_that_fails_gives_no_examples_and_the_rest_are_kept():
    graph = AsyncMock()
    graph.ro_query.side_effect = ResponseError("Query timed out")

    assert await _first_values(graph, "MATCH (n) RETURN n", 3) == ()


def test_a_string_with_a_backslash_or_a_newline_is_written_so_it_is_one_line():
    assert _literal("a\\b\nc") == "'a\\\\b\\nc'"


async def test_with_zero_nothing_is_read(
    falkordb_client: FalkorDB, film_graph: AsyncGraph, commands
):
    source = SchemaSource(falkordb_client, None, 0)
    await source.read(film_graph.name)
    without = sum(commands.values())
    commands.clear()

    schema, _ = await SchemaSource(falkordb_client, None, 3).read(film_graph.name)

    assert sum(commands.values()) > without
    assert schema.node_examples["Movie"] == {
        "title": ("'Matrix'",),
        "year": ("1999",),
        "rating": ("8.7",),
    }
    assert (await source.read(film_graph.name))[0].node_examples == {}


async def test_the_examples_are_kept_while_the_structure_of_the_graph_is_the_same(
    falkordb_client: FalkorDB, film_graph: AsyncGraph, commands
):
    source = SchemaSource(falkordb_client, None, 3)
    first, _ = await source.read(film_graph.name)
    commands.clear()
    await source.read(film_graph.name)
    again = sum(commands.values())
    commands.clear()
    await SchemaSource(falkordb_client, None, 0).read(film_graph.name)

    # a second read costs what a read with no examples costs: only the snapshot
    assert again == sum(commands.values())
    # values of the same structure that change are not read again
    await film_graph.query("MATCH (m:Movie) SET m.title = 'Reloaded'")
    kept, _ = await source.read(film_graph.name)
    assert kept.node_examples == first.node_examples


async def test_the_examples_are_read_again_when_the_structure_changes(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    source = SchemaSource(falkordb_client, None, 3)
    before, _ = await source.read(film_graph.name)
    await film_graph.query("MATCH (g:Genre) SET g.origin = 'France'")

    after, _ = await source.read(film_graph.name)

    assert "origin" not in before.node_examples["Genre"]
    assert after.node_examples["Genre"]["origin"] == ("'France'",)
