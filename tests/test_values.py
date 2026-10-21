import pickle
from collections import OrderedDict
from dataclasses import FrozenInstanceError

import pytest
from falkordb import Edge, Node, Path
from falkordb.asyncio import FalkorDB

from text2cypher import (
    Config,
    FakeLLM,
    GraphNode,
    GraphPath,
    GraphRelationship,
    Records,
    Text2Cypher,
    to_python,
)

pytestmark = pytest.mark.anyio


def keanu() -> Node:
    return Node(
        node_id=1, labels=["Person"], properties={"name": "Keanu", "born": 1964}
    )


def matrix() -> Node:
    return Node(node_id=2, labels=["Movie", "Film"], properties={"title": "Matrix"})


def acted() -> Edge:
    return Edge(1, "ACTED_IN", 2, edge_id=7, properties={"role": "Neo"})


def test_a_node_becomes_a_plain_object_with_id_labels_and_properties():
    node = to_python(keanu())

    assert isinstance(node, GraphNode)
    assert (node.id, node.labels) == (1, ("Person",))
    assert dict(node.properties) == {"name": "Keanu", "born": 1964}
    assert node["name"] == "Keanu"
    assert node.get("born") == 1964
    assert node.get("missing") is None
    assert node.get("missing", 0) == 0
    with pytest.raises(KeyError):
        node["missing"]


def test_a_node_has_several_labels_as_a_tuple_so_in_works():
    node = to_python(matrix())

    assert node.labels == ("Movie", "Film")
    assert "Film" in node.labels


def test_a_relationship_has_a_type_the_ids_of_its_ends_and_properties():
    edge = to_python(acted())

    assert isinstance(edge, GraphRelationship)
    assert (edge.id, edge.type, edge.start_id, edge.end_id) == (7, "ACTED_IN", 1, 2)
    assert edge["role"] == "Neo"
    assert edge.get("role") == "Neo"


def test_the_ends_of_a_relationship_that_are_nodes_are_read_as_their_ids():
    edge = to_python(Edge(keanu(), "ACTED_IN", matrix(), edge_id=7))

    assert (edge.start_id, edge.end_id) == (1, 2)


def test_a_path_has_its_nodes_and_relationships_in_order():
    path = to_python(Path([keanu(), matrix()], [acted()]))

    assert isinstance(path, GraphPath)
    assert [node.id for node in path.nodes] == [1, 2]
    assert [rel.type for rel in path.relationships] == ["ACTED_IN"]
    assert isinstance(path.nodes[0], GraphNode)


def test_nodes_inside_lists_maps_and_tuples_are_converted_and_the_rest_is_left_alone():
    value = {"people": [keanu(), "text", 3], "pair": (matrix(), None), "n": 1.5}
    ordered = OrderedDict(first=keanu())

    converted = to_python(value)

    assert converted["people"][0].id == 1
    assert converted["people"][1:] == ["text", 3]
    assert isinstance(converted["pair"], tuple)
    assert converted["pair"][0].id == 2
    assert converted["pair"][1] is None
    assert converted["n"] == 1.5
    assert type(to_python(ordered)) is dict
    assert to_python(ordered)["first"].labels == ("Person",)


def test_a_value_that_is_already_plain_comes_back_the_same():
    node = to_python(keanu())
    row = [node, {"a": [node]}, "x", None]

    assert to_python(row) == row
    assert to_python(node) is node
    assert to_python(to_python(keanu())) == node


def test_a_node_or_relationship_cannot_be_changed_and_its_properties_are_a_copy():
    properties = {"name": "Keanu"}
    node = GraphNode(1, ("Person",), properties)
    properties["name"] = "changed"

    assert node["name"] == "Keanu"
    with pytest.raises(FrozenInstanceError):
        node.id = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        node.properties["name"] = "x"  # ty: ignore[invalid-assignment]


def test_nodes_and_relationships_are_equal_and_hash_by_id():
    first = GraphNode(1, ("Person",), {"name": "a"})
    again = GraphNode(1, ("Person",), {"name": "a", "extra": 1})
    other = GraphNode(2, ("Person",), {"name": "a"})
    edge = GraphRelationship(1, "KNOWS", 1, 2, {})

    assert first == again
    assert hash(first) == hash(again)
    assert first != other
    assert len({first, again, other}) == 2
    assert edge != first
    assert edge == GraphRelationship(1, "LIKES", 5, 6, {})
    assert first != "node"


def test_the_repr_shows_everything_a_user_would_look_for():
    assert repr(to_python(keanu())) == (
        "GraphNode(id=1, labels=('Person',), properties={'name': 'Keanu', 'born': 1964})"
    )
    assert repr(to_python(acted())) == (
        "GraphRelationship(id=7, type='ACTED_IN', start_id=1, end_id=2, "
        "properties={'role': 'Neo'})"
    )


def test_the_objects_can_be_pickled_and_copied():
    node = to_python(keanu())

    again = pickle.loads(pickle.dumps(node))

    assert again == node
    assert again.labels == node.labels
    assert dict(again.properties) == dict(node.properties)


def test_as_python_converts_every_row_and_leaves_the_records_as_they_were():
    records = Records(("p", "name", "ms"), ((keanu(), "Keanu", [matrix()]),))

    converted = records.as_python()

    assert converted.columns == ("p", "name", "ms")
    assert isinstance(converted.rows[0][0], GraphNode)
    assert converted.rows[0][1] == "Keanu"
    assert converted.rows[0][2][0].labels == ("Movie", "Film")
    assert isinstance(records.rows[0][0], Node)
    assert converted.as_python() == converted
    assert Records((), ()).as_python() == Records((), ())


async def test_a_query_that_returns_nodes_gives_plain_objects_on_the_result(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            "```cypher\nMATCH (p:Person)-[r:ACTED_IN]->(m:Movie) RETURN p, r, m, p.name\n```"
        ]
    )

    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "Who acted in what?", movies_graph
    )

    assert result.records is not None
    assert isinstance(result.records.rows[0][0], Node)
    person, role, movie, name = result.records.as_python().rows[0]
    assert (person.labels, person["name"], name) == (("Person",), "Keanu", "Keanu")
    assert (role.type, role["role"]) == ("ACTED_IN", "Neo")
    assert (movie.labels, movie["title"], movie["year"]) == (("Movie",), "Matrix", 1999)
    assert (role.start_id, role.end_id) == (person.id, movie.id)


async def test_a_query_that_returns_paths_and_lists_of_nodes_gives_plain_objects(
    falkordb_client: FalkorDB, movies_graph: str
):
    llm = FakeLLM(
        [
            (
                "```cypher\nMATCH path=(p:Person)-[:ACTED_IN]->(m:Movie) "
                "RETURN path, collect(m) AS movies, {who: p} AS info\n```"
            )
        ]
    )

    result = await Text2Cypher(falkordb_client, llm, config=Config(execute=True)).run(
        "Show the path.", movies_graph
    )

    assert result.records is not None
    path, movies, info = result.records.as_python().rows[0]
    assert isinstance(path, GraphPath)
    assert [node.labels for node in path.nodes] == [("Person",), ("Movie",)]
    assert [rel.type for rel in path.relationships] == ["ACTED_IN"]
    assert [movie["title"] for movie in movies] == ["Matrix"]
    assert info["who"]["name"] == "Keanu"
