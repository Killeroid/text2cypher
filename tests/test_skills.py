import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from text2cypher._skills import Skill, load, skills

pytestmark = pytest.mark.anyio

EXAMPLE = re.compile(r"```cypher\n(.*?)\n```", re.DOTALL)
EXAMPLES = [
    pytest.param(skill.id, number, query, id=f"{skill.id}-{number}")
    for skill in skills().values()
    for number, query in enumerate(EXAMPLE.findall(skill.body), start=1)
]

# What the guides are written against: people who know each other, and cities joined by roads
GRAPH = """
CREATE (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'}), (carol:Person {name: 'Carol'}),
       (dave:Person {name: 'Dave'}), (eve:Person {name: 'Eve'}),
       (alice)-[:KNOWS]->(bob), (bob)-[:KNOWS]->(carol), (alice)-[:KNOWS]->(carol),
       (carol)-[:KNOWS]->(dave), (eve)-[:KNOWS]->(dave),
       (a:City {name: 'A', pop: 10}), (b:City {name: 'B', pop: 20}), (c:City {name: 'C', pop: 30}),
       (d:City {name: 'D', pop: 40}), (e:City {name: 'E', pop: 50}),
       (a)-[:ROAD {dist: 1, time: 5}]->(b), (b)-[:ROAD {dist: 1, time: 5}]->(c),
       (a)-[:ROAD {dist: 5, time: 1}]->(c), (c)-[:ROAD {dist: 2, time: 2}]->(d),
       (e)-[:ROAD {dist: 1, time: 1}]->(d)
"""


@pytest.fixture
async def guide_graph(falkordb_client: FalkorDB) -> AsyncIterator[AsyncGraph]:
    graph = falkordb_client.select_graph(f"skills_{uuid.uuid4().hex[:8]}")
    await graph.query(GRAPH)
    yield graph
    await graph.delete()


def test_the_skills_that_ship_are_these_in_id_order():
    assert list(skills()) == ["functions", "graph-algorithms", "path-finding"]
    for skill_id, skill in skills().items():
        assert skill.id == skill_id


def test_every_skill_has_a_name_a_one_line_description_and_examples_and_is_short():
    for skill in skills().values():
        assert skill.name
        assert skill.description
        assert "\n" not in skill.description
        # the only thing a model sees before it decides to read the skill: when, and what is in it
        assert skill.description.startswith("Use "), skill.id
        assert 100 < len(skill.description) < 300, (skill.id, len(skill.description))
        assert EXAMPLE.search(skill.body), skill.id
        # when to read it, then examples, then what to watch for
        headings = re.findall(r"^## (.+)$", skill.body, re.MULTILINE)
        assert headings == ["Usage", "Examples", "Notes"], (skill.id, headings)
        # reading one costs tokens in every request that reads it: about 300 to 800
        assert len(skill.text) <= 3000, (skill.id, len(skill.text))


def test_what_a_model_reads_is_the_name_and_the_guide():
    skill = skills()["path-finding"]

    assert skill.text.startswith("# Path finding\n\n## Usage\n")
    assert skill.text.endswith("only in `MATCH`.")


@pytest.mark.parametrize(("skill_id", "number", "query"), EXAMPLES)
async def test_every_example_runs_read_only_on_falkordb_and_returns_rows(
    guide_graph: AsyncGraph, skill_id: str, number: int, query: str
):
    result = await guide_graph.ro_query(query)

    assert result.result_set, f"{skill_id} example {number} returned nothing"


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (c:City) WHERE c.pop != 10 RETURN c",
        "MATCH (c:City) WHERE c.name =~ 'A.*' RETURN c",
        "MATCH (c:City {name: 'A'}) RETURN COUNT { (c)-[:ROAD]->() }",
        "MATCH (c:City {name: 'A'}) WHERE EXISTS { (c)-[:ROAD]->() } RETURN c",
        "RETURN datetime()",
        "RETURN toLowerCase('A')",
        "RETURN apoc.text.join(['a'], ',')",
        "MATCH p = shortestPath((a:City)-[:ROAD*]->(b:City)) RETURN p",
        "MATCH (a:City) WITH allShortestPaths((a)-[:ROAD*]->(a)) AS p RETURN p",
    ],
)
async def test_what_the_functions_skill_says_is_missing_really_is(
    guide_graph: AsyncGraph, query: str
):
    # if FalkorDB adds one, this fails and the skill needs its text changed
    with pytest.raises(Exception, match=r"."):
        await guide_graph.ro_query(query)


def test_a_skill_file_needs_front_matter_a_name_and_a_description(tmp_path: Path):
    for folder, text in {
        "none": "just text",
        "nameless": "---\ndescription: d\n---\nbody",
        "blank": "---\nname: n\ndescription:\n---\nbody",
    }.items():
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "skill.md").write_text(text)

        with pytest.raises(ValueError, match=folder):
            load(tmp_path)
        (tmp_path / folder / "skill.md").unlink()
        (tmp_path / folder).rmdir()


def test_the_loader_reads_a_folder_that_has_a_skill_and_skips_the_rest(tmp_path: Path):
    (tmp_path / "b-skill").mkdir()
    (tmp_path / "b-skill" / "skill.md").write_text(
        "---\nname: Bee\ndescription: About bees: a lot\n---\n\n# Heading\n\nText\n"
    )
    (tmp_path / "a-skill").mkdir()
    (tmp_path / "a-skill" / "skill.md").write_text(
        "---\nname: Ay\ndescription: About a\n---\nBody"
    )
    (tmp_path / "empty-folder").mkdir()
    (tmp_path / "file.txt").write_text("x")

    loaded = load(tmp_path)

    assert list(loaded) == ["a-skill", "b-skill"]
    assert loaded["b-skill"] == Skill(
        "b-skill", "Bee", "About bees: a lot", "# Heading\n\nText"
    )
    assert loaded["a-skill"].text == "# Ay\n\nBody"


async def rows(graph: AsyncGraph, query: str) -> list:
    return (await graph.ro_query(query)).result_set


async def test_the_notes_of_graph_algorithms_say_what_the_server_does(
    guide_graph: AsyncGraph,
):
    # PageRank takes two strings, not a map
    with pytest.raises(Exception, match="requires 2 arguments"):
        await rows(
            guide_graph,
            "CALL algo.pageRank({nodeLabels: ['Person']}) YIELD node RETURN node",
        )
    # the optional map: with none, the whole graph
    assert await rows(
        guide_graph, "CALL algo.betweenness() YIELD node, score RETURN count(node)"
    ) == [[10]]
    # WCC ignores direction: Eve only has an outgoing relationship, and still is in the one component
    assert await rows(
        guide_graph,
        "CALL algo.WCC({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) YIELD node, componentId RETURN count(DISTINCT componentId)",
    ) == [[1]]
    # BFS leaves out the start, and the depth and the type can be -1 and null
    reached = "MATCH (p:Person {name: 'Alice'}) CALL algo.BFS(p, %s) YIELD nodes RETURN [n IN nodes | n.name]"
    for arguments in ("2, 'KNOWS'", "-1, 'KNOWS'", "2, null"):
        [[names]] = await rows(guide_graph, reached % arguments)
        assert sorted(names) == ["Bob", "Carol", "Dave"], arguments


async def test_the_notes_of_path_finding_say_what_the_server_does(
    guide_graph: AsyncGraph,
):
    between = (
        "MATCH (a:City {name: 'A'}), (d:City {name: 'D'}) "
        "CALL algo.SPpaths({sourceNode: a, targetNode: d, relTypes: ['ROAD'], %s}) YIELD path "
        "RETURN [n IN nodes(path) | n.name]"
    )
    # one is the default, n is up to n, and 0 is those that tie for shortest
    assert await rows(guide_graph, between % "pathCount: 1") == [[["A", "C", "D"]]]
    assert len(await rows(guide_graph, between % "pathCount: 5")) == 2
    assert await rows(guide_graph, between % "pathCount: 0") == [[["A", "C", "D"]]]
    # maxLen bounds the hops
    assert await rows(guide_graph, between % "maxLen: 1") == []
    # relDirection: incoming walks the relationships backwards
    backwards = (
        "MATCH (a:City {name: 'A'}), (d:City {name: 'D'}) "
        "CALL algo.SPpaths({sourceNode: d, targetNode: a, relDirection: 'incoming'}) YIELD path "
        "RETURN [n IN nodes(path) | n.name]"
    )
    assert await rows(guide_graph, backwards) == [[["D", "C", "A"]]]
    # SSpaths: up to pathCount best paths overall, and not one per destination
    from_a = (
        "MATCH (a:City {name: 'A'}) CALL algo.SSpaths({sourceNode: a, relTypes: ['ROAD'], %s}) "
        "YIELD path RETURN count(path)"
    )
    assert await rows(guide_graph, from_a % "pathCount: 10") == [[5]]
    assert await rows(guide_graph, from_a % "pathCount: 3") == [[3]]
    # 0 is those that tie for the best: two one-hop paths unweighted, one by distance
    assert await rows(guide_graph, from_a % "pathCount: 0") == [[2]]
    assert await rows(guide_graph, from_a % "pathCount: 0, weightProp: 'dist'") == [[1]]
    # and the default is one
    assert await rows(guide_graph, from_a % "weightProp: 'dist'") == [[1]]
    # maxLen bounds the hops: only the two one-hop paths
    assert await rows(guide_graph, from_a % "pathCount: 10, maxLen: 1") == [[2]]
    # weightProp adds pathWeight, and costProp with maxCost adds pathCost and bounds it
    costed = (
        "MATCH (a:City {name: 'A'}) CALL algo.SSpaths({sourceNode: a, relTypes: ['ROAD'], "
        "pathCount: 10, weightProp: 'dist', costProp: 'time', maxCost: 6}) "
        "YIELD path, pathWeight, pathCost RETURN max(pathCost)"
    )
    [[most]] = await rows(guide_graph, costed)
    assert most <= 6


async def test_the_notes_of_the_functions_skill_say_what_the_server_does(
    guide_graph: AsyncGraph,
):
    # the optional fourth argument of list.insert skips a value that is already there
    assert await rows(guide_graph, "RETURN list.insert([1, 2, 3], 1, 2)") == [
        [[1, 2, 2, 3]]
    ]
    assert await rows(guide_graph, "RETURN list.insert([1, 2, 3], 1, 2, false)") == [
        [[1, 2, 3]]
    ]
    # remove takes a count from an index, and sort takes false for descending
    assert await rows(guide_graph, "RETURN list.remove([1, 2, 3, 4], 1, 2)") == [
        [[1, 4]]
    ]
    assert await rows(guide_graph, "RETURN list.sort([3, 1, 2], false)") == [
        [[3, 2, 1]]
    ]
    # the other spellings the notes give
    assert await rows(guide_graph, "RETURN toLower('AB')") == [["ab"]]
    assert await rows(
        guide_graph,
        "MATCH (c:City {name: 'A'}) RETURN size((c)-[:ROAD]->()), outdegree(c), indegree(c)",
    ) == [[2, 2, 0]]
    for substitute in ("localdatetime()", "date()", "timestamp()"):
        assert (await rows(guide_graph, f"RETURN {substitute} IS NOT NULL")) == [[True]]
    assert await rows(
        guide_graph,
        "MATCH (c:City) WHERE c.name STARTS WITH 'A' OR c.name ENDS WITH 'E' OR c.name CONTAINS 'C' RETURN count(c)",
    ) == [[3]]
