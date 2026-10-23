"""The runnable examples keep working: the documentation includes them."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.db  # it starts its own embedded database
def test_the_quickstart_example_runs_and_shows_the_correction():
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert "status: valid" in done.stdout
    assert "opposite direction" in done.stdout  # the validator's finding on attempt 1
    assert "attempt 2: MATCH (p:Person)-[:ACTED_IN]->" in done.stdout
    assert "Keanu Reeves" in done.stdout


QUERIES = {
    "How many movies are in the database?": "MATCH (m:Movie) RETURN count(m) AS movies",
    "List the titles of the movies released after 2000, newest first.": (
        "MATCH (m:Movie) WHERE m.year > 2000 RETURN m.title ORDER BY m.year DESC"
    ),
    "Which genre has the most movies, and how many?": (
        "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre) RETURN g.name, count(m) AS movies "
        "ORDER BY movies DESC LIMIT 1"
    ),
    "Show me the movie Se7en.": "MATCH (m:Movie {title: 'Se7en'}) RETURN m",
    "Show the people who were born in the United Kingdom or the Netherlands.": (
        "MATCH (p:Person)-[:BORN_IN]->(c:Country) "
        "WHERE c.name IN ['United Kingdom', 'Netherlands'] RETURN p"
    ),
    "Show the movies Keanu Reeves acted in.": (
        "MATCH (:Person {name: 'Keanu Reeves'})-[:ACTED_IN]->(m:Movie) RETURN m"
    ),
    "Show Keanu Reeves, the relationship, and the movie, for the movies he acted in.": (
        "MATCH (p:Person {name: 'Keanu Reeves'})-[r:ACTED_IN]->(m:Movie) RETURN p, r, m"
    ),
    (
        "What is the shortest chain of co-stars from Keanu Reeves to Brad Pitt? "
        "Return the path."
    ): (
        "MATCH (a:Person {name: 'Keanu Reeves'}), (b:Person {name: 'Brad Pitt'}) "
        "CALL algo.SPpaths({sourceNode: a, targetNode: b, relTypes: ['ACTED_IN'], "
        "relDirection: 'both', pathCount: 1}) YIELD path RETURN path"
    ),
    (
        "For each movie released before 2000, give the movie and the list of people "
        "who acted in it."
    ): (
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.year < 2000 "
        "RETURN m, collect(p) AS cast"
    ),
    "What is the average rating of each movie?": (
        "MATCH (:User)-[r:REVIEWED]->(m:Movie) RETURN m.title, avg(r.stars) AS rating "
        "ORDER BY rating DESC"
    ),
    "Delete every movie from before 1990.": "",
    "Which actors appeared in The Matrix?": (
        "MATCH (p:Person)-[:ACTED_IN]->(:Movie {title: 'The Matrix'}) RETURN p.name"
    ),
    "How many people were born in the United States?": (
        "MATCH (:Person)-[:BORN_IN]->(:Country {name: 'United States'}) "
        "RETURN count(*) AS people"
    ),
    "Which studio produced the most movies?": (
        "MATCH (m:Movie)-[:PRODUCED_BY]->(s:Studio) RETURN s.name, count(m) AS movies "
        "ORDER BY movies DESC LIMIT 1"
    ),
    "Who directed Gravity?": (
        "MATCH (p:Person)-[:DIRECTED]->(:Movie {title: 'Gravity'}) RETURN p.name"
    ),
    "Who directed Cast Away?": (
        "MATCH (p:Person)-[:DIRECTED]->(:Movie {title: 'Cast Away'}) RETURN p.name"
    ),
}
# The questions whose answer starts with the model reading a skill
READS = {
    "What is the shortest chain of co-stars from Keanu Reeves to Brad Pitt? "
    "Return the path.": "path-finding",
    "What is the average rating of each movie?": "film-notes",
}


class ScriptedModel:
    """Answers the questions of the tour the way a model would: it keeps the whole
    schema when asked to filter, reads a skill for two questions, and writes the query."""

    supports_tools = True

    def __init__(self) -> None:
        self.reads = 0

    async def chat(self, system, messages, tools=None, *, tool_choice="auto"):
        from text2cypher import Completion, ToolCall

        if "You choose which parts of a graph database's schema" in system:
            return Completion(
                '```json\n{"labels": ["Person", "Movie", "Genre", "Studio", "Country", '
                '"User"], "relationship_types": ["ACTED_IN", "DIRECTED", "IN_GENRE", '
                '"PRODUCED_BY", "BORN_IN", "REVIEWED", "FOLLOWS"]}\n```',
                None,
            )
        question = next(
            m.content[len("# Question\n") :]
            for m in reversed(messages)
            if m.role == "user" and m.content.startswith("# Question\n")
        )
        skill = READS.get(question)
        if skill and tools and not any(m.role == "tool" for m in messages):
            self.reads += 1
            return Completion("", None, (ToolCall("c1", "read_skill", {"id": skill}),))
        return Completion(f"```cypher\n{QUERIES[question]}\n```", None)


def load_tour():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "full_tour", EXAMPLES / "full_tour.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.anyio
async def test_the_full_tour_runs_every_section_with_a_scripted_model(
    falkordb_client, capsys
):
    tour = load_tour()
    model = ScriptedModel()

    await tour.tour(falkordb_client, model, model=None, filter_model=None)
    out = capsys.readouterr().out

    assert tour.GRAPH not in await falkordb_client.list_graphs()  # cleaned up
    assert model.reads == 2
    for section in range(1, 8):
        assert f"\n{section}. " in out
    # values, one node, nodes of my own classes, a relationship, a path, nested nodes
    assert "row: (7,)" in out
    assert "as an object: node #" in out
    assert "as my own class: [Person(name='Jan de Bont', born=1943)]" in out
    assert "Movie(title='The Matrix', year=1999)" in out
    assert "relationship #" in out
    assert "ACTED_IN" in out
    assert (
        "path Keanu Reeves -> Speed -> Sandra Bullock -> Gravity -> George Clooney "
        "-> Ocean's Eleven -> Brad Pitt"
    ) in out
    assert "The Matrix | ['Keanu Reeves', 'Carrie-Anne Moss']" in out
    assert "read skills: path-finding" in out
    assert "read skills: film-notes" in out
    # declined, query only, errors
    assert "status: unanswerable" in out
    assert "(declined: no query)" in out
    assert "records: None (nothing was run)" in out
    assert "no such graph" in out
    assert "Who directed Cast Away? -> valid" in out
    assert "status: failed" not in out
