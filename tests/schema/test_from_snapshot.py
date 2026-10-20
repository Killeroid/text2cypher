from collections.abc import Awaitable, Callable

import pytest
from cypher_guardrail import Snapshot
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from text2cypher.schema import GraphSchema, Index, Path, schema_from_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def schema_of(
    falkordb_client: FalkorDB,
) -> Callable[[AsyncGraph], Awaitable[GraphSchema]]:
    async def schema_of(graph: AsyncGraph) -> GraphSchema:
        snapshot = await Snapshot.capture_async(falkordb_client, graph.name, types=True)
        return schema_from_snapshot(snapshot)

    return schema_of


async def test_schema_has_properties_types_and_paths(film_graph: AsyncGraph, schema_of):
    schema = await schema_of(film_graph)

    person = {
        "alive": ("Boolean",),
        "born": ("Integer",),
        "name": ("String",),
        "tags": ("List",),
    }
    assert schema == GraphSchema(
        node_properties={
            # Keanu is both a Person and an Actor
            "Actor": person,
            "Genre": {"name": ("String",)},
            "Movie": {
                "rating": ("Float",),
                "title": ("String",),
                "year": ("Integer",),
            },
            "Person": person,
        },
        relationship_properties={
            "ACTED_IN": {"role": ("String",), "since": ("Integer",)},
            "IN_GENRE": {},
            "KNOWS": {},
        },
        paths=(
            Path("Actor", "ACTED_IN", "Movie"),
            Path("Actor", "KNOWS", "Person"),
            Path("Movie", "IN_GENRE", "Genre"),
            Path("Person", "ACTED_IN", "Movie"),
            Path("Person", "KNOWS", "Person"),
        ),
    )


async def test_graph_with_no_data_has_an_empty_schema(new_graph: AsyncGraph, schema_of):
    assert await schema_of(new_graph) == GraphSchema({}, {}, ())


async def test_a_property_with_several_value_types_lists_all_of_them(
    new_graph: AsyncGraph, schema_of
):
    await new_graph.query("CREATE (:Thing {size: 1}), (:Thing {size: 'big'})")

    schema = await schema_of(new_graph)

    assert schema.node_properties == {"Thing": {"size": ("Integer", "String")}}


async def test_labels_and_properties_that_need_quoting_are_found(
    new_graph: AsyncGraph, schema_of
):
    await new_graph.query(
        "CREATE (a:`My Label` {`a b`: 1})-[:`rel type`]->(b:`other one`)"
    )

    schema = await schema_of(new_graph)

    assert schema.node_properties == {
        "My Label": {"a b": ("Integer",)},
        "other one": {},
    }
    assert schema.paths == (Path("My Label", "rel type", "other one"),)


async def test_a_rarer_path_of_the_same_type_is_found(new_graph: AsyncGraph, schema_of):
    await new_graph.query(
        "CREATE (a:Person), (b:Person), (c:Person), (x:Company), "
        "(a)-[:LINKED]->(b), (a)-[:LINKED]->(c), (b)-[:LINKED]->(c), "
        "(c)-[:LINKED]->(x)"
    )

    schema = await schema_of(new_graph)

    assert schema.paths == (
        Path("Person", "LINKED", "Company"),
        Path("Person", "LINKED", "Person"),
    )


def test_names_with_a_backtick_are_left_out():
    # A graph can hold such a name (e.g. from a bulk load) though FalkorDB cannot parse it
    snapshot = Snapshot.from_schema(
        {"Good": ["fine", "evil`prop"], "evil`label": []},
        [
            ("Good", "REL", "Good"),
            ("Good", "REL", "evil`label"),
            ("Good", "evil`type", "Good"),
        ],
        relationship_properties={"REL": [], "evil`type": []},
        node_property_types={"Good": {"fine": ["String"], "evil`prop": ["String"]}},
        relationship_property_types={"REL": {}, "evil`type": {}},
    )

    schema = schema_from_snapshot(snapshot)

    assert schema == GraphSchema(
        node_properties={"Good": {"fine": ("String",)}},
        relationship_properties={"REL": {}},
        paths=(Path("Good", "REL", "Good"),),
    )


async def test_a_snapshot_without_value_types_is_rejected(
    falkordb_client: FalkorDB, film_graph: AsyncGraph
):
    snapshot = await Snapshot.capture_async(falkordb_client, film_graph.name)

    with pytest.raises(ValueError, match="types=True"):
        schema_from_snapshot(snapshot)


def test_the_indexes_of_a_snapshot_are_in_the_schema():
    snapshot = Snapshot.from_schema(
        nodes={"Person": ["name", "born"], "Movie": ["title"], "Doc": ["embedding"]},
        relationships=[("Person", "ACTED_IN", "Movie")],
        relationship_properties={"ACTED_IN": ["role"]},
        indexes=[
            ("Person", "NODE", {"name": ["RANGE"], "born": ["RANGE"]}),
            ("Movie", "NODE", {"title": ["FULLTEXT", "RANGE"]}),
            ("Doc", "NODE", {"embedding": ["VECTOR"]}),
            ("ACTED_IN", "RELATIONSHIP", {"role": ["RANGE"]}),
        ],
        node_property_types={
            "Person": {"name": ["String"], "born": ["Integer"]},
            "Movie": {"title": ["String"]},
            "Doc": {"embedding": ["Vectorf32"]},
        },
        relationship_property_types={"ACTED_IN": {"role": ["String"]}},
    )

    schema = schema_from_snapshot(snapshot)

    assert set(schema.indexes) == {
        Index("Person", "name", "RANGE"),
        Index("Person", "born", "RANGE"),
        Index("Movie", "title", "FULLTEXT"),
        Index("Movie", "title", "RANGE"),
        Index("Doc", "embedding", "VECTOR"),
        Index("ACTED_IN", "role", "RANGE"),
    }
    assert list(schema.indexes) == sorted(schema.indexes)


def test_an_index_on_a_name_with_a_backtick_is_left_out_like_the_name():
    snapshot = Snapshot.from_schema(
        nodes={"Odd`Label": ["x"], "Fine": ["y"]},
        relationships=[],
        indexes=[
            ("Odd`Label", "NODE", {"x": ["RANGE"]}),
            ("Fine", "NODE", {"y": ["RANGE"]}),
        ],
        node_property_types={"Odd`Label": {"x": ["String"]}, "Fine": {"y": ["String"]}},
    )

    assert schema_from_snapshot(snapshot).indexes == (Index("Fine", "y", "RANGE"),)


def test_the_user_defined_functions_of_a_snapshot_are_in_the_schema():
    snapshot = Snapshot.from_schema(
        nodes={"A": ["x"]},
        relationships=[],
        node_property_types={"A": {"x": ["String"]}},
        udfs={"b": ["g"], "a": ["f", "e"]},
    )

    assert schema_from_snapshot(snapshot).functions == ("a.e", "a.f", "b.g")


def test_a_snapshot_that_did_not_capture_the_functions_gives_none():
    snapshot = Snapshot.from_schema(
        nodes={"A": ["x"]},
        relationships=[],
        node_property_types={"A": {"x": ["String"]}},
    )

    assert schema_from_snapshot(snapshot).functions == ()
