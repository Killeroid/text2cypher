"""A tour of text2cypher with the OpenAI SDK, against a FalkorDB server.

It seeds a small film graph and asks it questions in plain English, so you can see each
kind of answer: a single value, a list of values, one node, a list of nodes, a
relationship, a path, nodes inside lists, a question that cannot be answered, and the
same question without running the query. It shows the options that matter along the way:

* the OpenAI SDK's own client with ``model="gpt-4o"``;
* an LLM schema filter that asks a cheaper model which labels a question needs;
* skills the model reads as a tool (the ones that ship, and one of your own);
* execution, example values, typo fixing, a kept schema and limits for the correction loop;
* nodes, relationships and paths as plain Python objects, and how to make your own
  classes from them.

You need a FalkorDB server, an OpenAI API key, and the OpenAI SDK (``uv add openai``):

    docker run -p 6379:6379 -d falkordb/falkordb
    export OPENAI_API_KEY=...
    uv run python examples/full_tour.py

Set ``FALKORDB_HOST`` and ``FALKORDB_PORT`` if the server is not on localhost:6379. The
tour makes a graph called ``text2cypher_tour`` and deletes it at the end. Everything the
model is sent (the schema, example values from this graph) goes to OpenAI.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Self

from falkordb.asyncio import FalkorDB
from openai import AsyncOpenAI

from text2cypher import (
    AsyncChatClient,
    Config,
    DatabaseError,
    GraphNode,
    GraphNotFoundError,
    GraphPath,
    GraphRelationship,
    LLMError,
    LLMSchemaFilter,
    Result,
    Skill,
    Text2Cypher,
    to_python,
)
from text2cypher.skills import builtin_skills

GRAPH = "text2cypher_tour"

# Six labels and seven relationship types. People act in and direct movies; movies have
# genres, studios and countries; users review movies and follow each other. Keanu Reeves
# and Brad Pitt are linked by a chain of co-stars (Speed, Gravity, Ocean's Eleven).
SEED = """
CREATE (usa:Country {name: 'United States'}), (uk:Country {name: 'United Kingdom'}),
       (mexico:Country {name: 'Mexico'}), (netherlands:Country {name: 'Netherlands'}),
       (action:Genre {name: 'Action'}), (scifi:Genre {name: 'Sci-Fi'}),
       (thriller:Genre {name: 'Thriller'}), (drama:Genre {name: 'Drama'}),
       (crime:Genre {name: 'Crime'}),
       (warner:Studio {name: 'Warner Bros.'}), (fox:Studio {name: '20th Century Fox'}),
       (universal:Studio {name: 'Universal Pictures'}),
       (keanu:Person {name: 'Keanu Reeves', born: 1964}),
       (carrie:Person {name: 'Carrie-Anne Moss', born: 1967}),
       (sandra:Person {name: 'Sandra Bullock', born: 1964}),
       (george:Person {name: 'George Clooney', born: 1961}),
       (brad:Person {name: 'Brad Pitt', born: 1963}),
       (morgan:Person {name: 'Morgan Freeman', born: 1937}),
       (tom:Person {name: 'Tom Hanks', born: 1956}),
       (kevin:Person {name: 'Kevin Bacon', born: 1958}),
       (lana:Person {name: 'Lana Wachowski', born: 1965}),
       (jan:Person {name: 'Jan de Bont', born: 1943}),
       (alfonso:Person {name: 'Alfonso Cuaron', born: 1961}),
       (steven:Person {name: 'Steven Soderbergh', born: 1963}),
       (david:Person {name: 'David Fincher', born: 1962}),
       (ron:Person {name: 'Ron Howard', born: 1954}),
       (matrix:Movie {title: 'The Matrix', year: 1999}),
       (speed:Movie {title: 'Speed', year: 1994}),
       (gravity:Movie {title: 'Gravity', year: 2013}),
       (ocean:Movie {title: "Ocean's Eleven", year: 2001}),
       (seven:Movie {title: 'Se7en', year: 1995}),
       (apollo:Movie {title: 'Apollo 13', year: 1995}),
       (castaway:Movie {title: 'Cast Away', year: 2000}),
       (keanu)-[:ACTED_IN {role: 'Neo'}]->(matrix),
       (carrie)-[:ACTED_IN {role: 'Trinity'}]->(matrix),
       (keanu)-[:ACTED_IN {role: 'Jack Traven'}]->(speed),
       (sandra)-[:ACTED_IN {role: 'Annie Porter'}]->(speed),
       (sandra)-[:ACTED_IN {role: 'Ryan Stone'}]->(gravity),
       (george)-[:ACTED_IN {role: 'Matt Kowalski'}]->(gravity),
       (george)-[:ACTED_IN {role: 'Danny Ocean'}]->(ocean),
       (brad)-[:ACTED_IN {role: 'Rusty Ryan'}]->(ocean),
       (brad)-[:ACTED_IN {role: 'David Mills'}]->(seven),
       (morgan)-[:ACTED_IN {role: 'William Somerset'}]->(seven),
       (tom)-[:ACTED_IN {role: 'Jim Lovell'}]->(apollo),
       (kevin)-[:ACTED_IN {role: 'Jack Swigert'}]->(apollo),
       (tom)-[:ACTED_IN {role: 'Chuck Noland'}]->(castaway),
       (lana)-[:DIRECTED]->(matrix), (jan)-[:DIRECTED]->(speed),
       (alfonso)-[:DIRECTED]->(gravity), (steven)-[:DIRECTED]->(ocean),
       (david)-[:DIRECTED]->(seven), (ron)-[:DIRECTED]->(apollo),
       (matrix)-[:IN_GENRE]->(action), (matrix)-[:IN_GENRE]->(scifi),
       (speed)-[:IN_GENRE]->(action), (speed)-[:IN_GENRE]->(thriller),
       (gravity)-[:IN_GENRE]->(scifi), (gravity)-[:IN_GENRE]->(thriller),
       (ocean)-[:IN_GENRE]->(crime), (seven)-[:IN_GENRE]->(thriller),
       (seven)-[:IN_GENRE]->(crime), (apollo)-[:IN_GENRE]->(drama),
       (castaway)-[:IN_GENRE]->(drama),
       (matrix)-[:PRODUCED_BY]->(warner), (ocean)-[:PRODUCED_BY]->(warner),
       (speed)-[:PRODUCED_BY]->(fox), (gravity)-[:PRODUCED_BY]->(warner),
       (apollo)-[:PRODUCED_BY]->(universal),
       (keanu)-[:BORN_IN]->(usa), (carrie)-[:BORN_IN]->(usa),
       (sandra)-[:BORN_IN]->(usa), (george)-[:BORN_IN]->(usa),
       (brad)-[:BORN_IN]->(usa), (morgan)-[:BORN_IN]->(usa),
       (tom)-[:BORN_IN]->(usa), (kevin)-[:BORN_IN]->(usa),
       (lana)-[:BORN_IN]->(usa), (jan)-[:BORN_IN]->(netherlands),
       (alfonso)-[:BORN_IN]->(mexico), (steven)-[:BORN_IN]->(usa),
       (david)-[:BORN_IN]->(usa), (ron)-[:BORN_IN]->(usa),
       (ana:User {name: 'Ana'}), (ben:User {name: 'Ben'}), (cy:User {name: 'Cy'}),
       (ana)-[:REVIEWED {stars: 5}]->(matrix), (ana)-[:REVIEWED {stars: 4}]->(seven),
       (ben)-[:REVIEWED {stars: 3}]->(matrix), (ben)-[:REVIEWED {stars: 5}]->(gravity),
       (cy)-[:REVIEWED {stars: 4}]->(apollo), (cy)-[:REVIEWED {stars: 2}]->(speed),
       (ana)-[:FOLLOWS]->(ben), (ben)-[:FOLLOWS]->(cy)
"""

# A skill of your own. A skill is a short guide the model can read when a question needs
# it; this one tells it how this graph is used. The description is all the model sees
# before it decides to read the body, so it says when to read it.
FILM_NOTES = Skill(
    id="film-notes",
    name="How this film graph is used",
    description=(
        "Use for questions about reviews, ratings, stars and who follows whom among "
        "the users of this film graph."
    ),
    body="""\
## Usage

Ratings are the `stars` property of a `REVIEWED` relationship, from 1 to 5. A movie's
rating is the average of its reviews: `avg(r.stars)`. Movies nobody has reviewed have none.

## Examples

```cypher
MATCH (:User)-[r:REVIEWED]->(m:Movie) RETURN m.title, avg(r.stars) AS rating ORDER BY rating DESC
```

## Notes

`FOLLOWS` goes from the follower to the one being followed.
""",
)


def build_agent(
    client: FalkorDB,
    llm: AsyncOpenAI | AsyncChatClient,
    *,
    model: str | None = "gpt-4o",
    filter_model: str | None = "gpt-4o-mini",
) -> Text2Cypher:
    """The agent the tour uses, with the options that matter spelled out.

    ``llm`` is the SDK's own client and ``model`` says which model writes the queries. An
    ``AsyncAzureOpenAI`` or a client for a compatible server (``base_url=...``) works the
    same way, and so does a client of your own with a ``chat`` method (see the LLM guide),
    which has its own model, so ``model`` and ``filter_model`` are ``None`` for it.
    """
    return Text2Cypher(
        client,
        llm,
        model=model,
        config=Config(
            # Run a query that passed validation, read-only, and correct it when FalkorDB
            # fails it. Without this you get the query and no rows.
            execute=True,
            execute_timeout_ms=5000,
            # Offer skills as a tool the model reads when a question needs them. Each
            # read is one more call to the model; a plain lookup reads nothing.
            use_skills=True,
            # Up to three example values of each property go into the schema, so the model
            # writes 'United States' and not 'USA'. They are rows of your graph and are
            # sent to OpenAI: 0 sends none.
            example_values=3,
            # Fix a misspelled label, relationship type or property without asking the
            # model again, when exactly one name of the graph is a likely typo of it.
            autofix_typos=True,
            # Keep the schema for five minutes. Call agent.clear_schema(GRAPH) after you
            # change the graph's structure sooner than that.
            schema_refresh_seconds=300,
            # The correction loop: 3 corrections with the part of the schema the filter
            # kept, then up to 6 in all, with the whole schema. Defaults are 5 and 10.
            attempts_w_filtered=3,
            total_attempts=6,
            # Other options: use_parameters=True (the model writes the values in a CYPHER
            # header, which FalkorDB can reuse a plan for) and check_performance=True
            # (reports a cartesian product, a scan of every node, a path with no bound;
            # off by default because a shortest path between two nodes is a product on
            # purpose). check_types is on by default.
        ),
        # Ask a model which labels and relationship types a question needs, instead of
        # matching its words to names. filter_model is a cheaper model on the same client.
        # A schema with fewer than min_entities names is kept whole without asking; the
        # default is 20, which a real graph has, and this small one lowers it to show it.
        schema_filter=LLMSchemaFilter(model=filter_model, min_entities=10),
        # The skills that ship (graph algorithms, path finding, functions) and one more.
        # Giving `skills=` replaces the shipped ones, so they are listed here.
        skills=[*builtin_skills(), FILM_NOTES],
    )


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(result: Result, *, rows: int = 5) -> None:
    """What came of a question: the status, each query, what it cost, and the rows."""
    print(f"\nQ: {result.question}")
    print(f"status: {result.status}")
    for number, attempt in enumerate(result.attempts, start=1):
        extra = []
        if attempt.skills:
            extra.append(f"read skills: {', '.join(attempt.skills)}")
        if attempt.auto_fixed:
            extra.append("typo fixed without the model")
        print(
            f"  attempt {number} ({attempt.schema_scope} schema"
            f"{'; ' + '; '.join(extra) if extra else ''}):"
        )
        print(f"    {' '.join(attempt.query.split()) or '(declined: no query)'}")
        for issue in attempt.validation.issues if attempt.validation else ():
            print(f"    problem: {issue.description}")
    if result.usage:
        print(
            f"  tokens: {result.usage.input_tokens} in, {result.usage.output_tokens} out"
            f" (the schema filter used {result.filter_usage})"
        )
    if result.records:
        print(f"  columns: {result.records.columns}")
        # show() prints the converted rows (as_python()), so nodes read GraphNode(...) and not
        # <falkordb.node.Node object at 0x...>; result.records.rows itself is not changed
        for row in result.records.as_python().rows[:rows]:
            print(f"  row: {row}")
        if len(result.records.rows) > rows:
            print(f"  ... {len(result.records.rows) - rows} more rows")
    if result.execution_error:
        print(f"  not run: {result.execution_error}")


def nodes_in(value: object) -> list[GraphNode]:
    """Every node in a converted value, also inside lists and maps."""
    if isinstance(value, GraphNode):
        return [value]
    if isinstance(value, list | tuple):
        return [node for item in value for node in nodes_in(item)]
    if isinstance(value, dict):
        return [node for item in value.values() for node in nodes_in(item)]
    return []


def nodes_of(result: Result, label: str) -> list[GraphNode]:
    """The nodes with ``label`` that a query returned, in any column.

    A model may return the nodes (``RETURN p``) or only their properties (``RETURN
    p.name``), depending on the question and on how it words the query. Only the first
    has nodes in it, so look before you build objects from them.
    """
    if result.records is None:
        return []
    found = [
        node
        for row in result.records.as_python().rows
        for cell in row
        for node in nodes_in(cell)
    ]
    return [node for node in found if label in node.labels]


def describe(value: object) -> str:
    """One line for a value that came from ``as_python()``."""
    if isinstance(value, GraphNode):
        return f"node #{value.id} {':'.join(value.labels)} {dict(value.properties)}"
    if isinstance(value, GraphRelationship):
        return (
            f"relationship #{value.id} {value.type} "
            f"({value.start_id} -> {value.end_id}) {dict(value.properties)}"
        )
    if isinstance(value, GraphPath):
        names = (node.get("name") or node.get("title") for node in value.nodes)
        return "path " + " -> ".join(map(str, names))
    if isinstance(value, list):
        return "[" + ", ".join(describe(item) for item in value) + "]"
    return repr(value)


def names(value: object) -> str:
    """The name or title of a node (or of each node in a list), or the value itself."""
    if isinstance(value, GraphNode):
        return str(value.get("name") or value.get("title"))
    if isinstance(value, list):
        return str([names(item) for item in value])
    return str(value)


# Your own classes, made from the plain objects. A mapper from an ORM could do the same
# job, but it usually wants the client's own node, which result.records.rows still holds.
@dataclass(frozen=True)
class Movie:
    title: str
    year: int

    @classmethod
    def from_node(cls, node: GraphNode) -> Self:
        if "Movie" not in node.labels:
            raise ValueError(f"not a movie: {node.labels}")
        return cls(node["title"], node["year"])


@dataclass(frozen=True)
class Person:
    name: str
    born: int | None = None

    @classmethod
    def from_node(cls, node: GraphNode) -> Self:
        return cls(node["name"], node.get("born"))


# What each section prints is shown in the comments, as it looks when the first query
# is right. A model words its queries its own way and may need a correction or two, in
# which case you see more attempts, and its token counts are its own.
#
# The first attempt uses the schema the LLM filter kept ("filtered schema"); a
# correction after the filtered attempts are used up, or a decline, moves to "full schema".


async def values(agent: Text2Cypher) -> None:
    heading("1. Values: a number, and lists of values")
    # The prompt asks for the properties a question wants (RETURN m.title), not whole nodes
    for question in (
        "How many movies are in the database?",
        "List the titles of the movies released after 2000, newest first.",
        "Which genre has the most movies, and how many?",
    ):
        show(await agent.run(question, graph=GRAPH))
    # Q: How many movies are in the database?
    # status: valid
    #   attempt 1 (filtered schema):
    #     MATCH (m:Movie) RETURN count(m) AS movies
    #   tokens: 1630 in, 24 out (the schema filter used Usage(input_tokens=402, output_tokens=18))
    #   columns: ('movies',)
    #   row: (7,)
    #
    # Q: List the titles of the movies released after 2000, newest first.
    # ...
    #   columns: ('m.title',)
    #   row: ('Gravity',)
    #   row: ("Ocean's Eleven",)
    #
    # Q: Which genre has the most movies, and how many?
    # ...
    #   columns: ('g.name', 'movies')
    #   row: ('Thriller', 3)


async def nodes(agent: Text2Cypher) -> None:
    heading("2. Nodes: one, and a list, as plain objects")
    # A question about "the movie" or "the people" themselves gets whole nodes. The rows
    # of result.records hold the falkordb client's own values; as_python() makes them plain
    # objects. show() calls as_python() itself when it prints the rows (see above), which is
    # why the row below reads GraphNode(...) although we have not converted anything here.
    result = await agent.run("Show me the movie Se7en.", graph=GRAPH)
    show(result)
    # Q: Show me the movie Se7en.
    # status: valid
    #   attempt 1 (filtered schema):
    #     MATCH (m:Movie {title: 'Se7en'}) RETURN m
    #   columns: ('m',)
    #   row: (GraphNode(id=30, labels=('Movie',), properties={'title': 'Se7en', 'year': 1995}),)
    if result.records and result.records.rows:
        # result.records.rows is still what the client returned, so this cell is its own Node
        raw = result.records.rows[0][0]
        print(
            f"  the raw cell is the client's own: {type(raw).__module__}.{type(raw).__name__}"
        )
        # the raw cell is the client's own: falkordb.node.Node
    for node in nodes_of(result, "Movie"):
        print(f"  as an object: {describe(node)}")
        print(f"  labels={node.labels} id={node.id} title={node['title']!r}")
        # as an object: node #30 Movie {'title': 'Se7en', 'year': 1995}
        # labels=('Movie',) id=30 title='Se7en'
    if not nodes_of(result, "Movie"):
        print("  the query returned values and not nodes: nothing to convert")

    result = await agent.run(
        "Show the people who were born in the United Kingdom or the Netherlands.",
        graph=GRAPH,
    )
    show(result)
    people = [Person.from_node(node) for node in nodes_of(result, "Person")]
    print(f"  as my own class: {people}")
    # ...
    #   row: (GraphNode(id=21, labels=('Person',), properties={'name': 'Jan de Bont', 'born': 1943}),)
    #   as my own class: [Person(name='Jan de Bont', born=1943)]

    result = await agent.run("Show the movies Keanu Reeves acted in.", graph=GRAPH)
    show(result)
    movies = [Movie.from_node(node) for node in nodes_of(result, "Movie")]
    print(f"  as my own class: {sorted(movies, key=lambda movie: movie.year)}")
    # ...
    #   row: (GraphNode(id=26, labels=('Movie',), properties={'title': 'The Matrix', 'year': 1999}),)
    #   row: (GraphNode(id=27, labels=('Movie',), properties={'title': 'Speed', 'year': 1994}),)
    #   as my own class: [Movie(title='Speed', year=1994), Movie(title='The Matrix', year=1999)]


async def relationships_and_paths(agent: Text2Cypher) -> None:
    heading("3. A relationship, and a path (the model reads the path-finding skill)")
    result = await agent.run(
        "Show Keanu Reeves, the relationship, and the movie, for the movies he acted in.",
        graph=GRAPH,
    )
    show(result, rows=2)
    if result.records:
        for row in result.records.as_python().rows[:2]:
            print("  " + " | ".join(describe(cell) for cell in row))
    # columns: ('p', 'r', 'm')
    # row: (GraphNode(id=12, ...), GraphRelationship(id=0, type='ACTED_IN', start_id=12, end_id=26, ...), GraphNode(id=26, ...))
    # node #12 Person {'name': 'Keanu Reeves', 'born': 1964} | relationship #0 ACTED_IN (12 -> 26) {'role': 'Neo'} | node #26 Movie {'title': 'The Matrix', 'year': 1999}
    # node #12 Person {...} | relationship #2 ACTED_IN (12 -> 27) {'role': 'Jack Traven'} | node #27 Movie {'title': 'Speed', 'year': 1994}

    result = await agent.run(
        "What is the shortest chain of co-stars from Keanu Reeves to Brad Pitt? "
        "Return the path.",
        graph=GRAPH,
    )
    show(result)
    # Q: What is the shortest chain of co-stars from Keanu Reeves to Brad Pitt? Return the path.
    # status: valid
    #   attempt 1 (filtered schema; read skills: path-finding):
    #     MATCH (a:Person {name: 'Keanu Reeves'}), (b:Person {name: 'Brad Pitt'}) CALL algo.SPpaths({...}) YIELD path RETURN path
    #   columns: ('path',)
    #   row: (GraphPath(nodes=(GraphNode(id=12, ...), ...), relationships=(...)),)
    if result.records and result.records.rows:
        # to_python converts one value, here the first cell of the first row
        print(f"  {describe(to_python(result.records.rows[0][0]))}")
        # path Keanu Reeves -> Speed -> Sandra Bullock -> Gravity -> George Clooney -> Ocean's Eleven -> Brad Pitt


async def nested(agent: Text2Cypher) -> None:
    heading("4. Nodes inside lists, and your own skill")
    # collect(...) makes a list of nodes in a cell; as_python() converts inside it
    result = await agent.run(
        "For each movie released before 2000, give the movie and the list of people "
        "who acted in it.",
        graph=GRAPH,
    )
    show(result, rows=2)
    if result.records:
        for row in result.records.as_python().rows[:3]:
            print("  " + " | ".join(names(cell) for cell in row))
    # columns: ('m', 'cast')
    # row: (GraphNode(id=31, labels=('Movie',), ...), [GraphNode(id=18, ...), GraphNode(id=19, ...)])
    # row: (GraphNode(id=26, labels=('Movie',), ...), [GraphNode(id=12, ...), GraphNode(id=13, ...)])
    # ... 2 more rows
    # Apollo 13 | ['Tom Hanks', 'Kevin Bacon']
    # The Matrix | ['Keanu Reeves', 'Carrie-Anne Moss']
    # Speed | ['Keanu Reeves', 'Sandra Bullock']

    # This one is answered with the help of the skill of our own (film-notes)
    show(await agent.run("What is the average rating of each movie?", graph=GRAPH))
    # attempt 1 (filtered schema; read skills: film-notes):
    #   MATCH (:User)-[r:REVIEWED]->(m:Movie) RETURN m.title, avg(r.stars) AS rating ORDER BY rating DESC
    # columns: ('m.title', 'rating')
    # row: ('Gravity', 5.0)
    # row: ('Apollo 13', 4.0)
    # row: ('Se7en', 4.0)
    # row: ('The Matrix', 4.0)
    # row: ('Speed', 2.0)


async def declined_and_query_only(agent: Text2Cypher, query_only: Text2Cypher) -> None:
    heading("5. A question that cannot be answered, and the query without running it")
    # A request to change data is declined, and a query that could write is never valid
    result = await agent.run("Delete every movie from before 1990.", graph=GRAPH)
    show(result)
    print(f"  (status {result.status}: the model declined, so there is no query)")
    # Q: Delete every movie from before 1990.
    # status: unanswerable
    #   attempt 1 (filtered schema):
    #     (declined: no query)
    #   attempt 2 (full schema):
    #     (declined: no query)
    #   (status unanswerable: the model declined, so there is no query)
    # A decline on a filtered schema is asked once more on the full one, in case the filter
    # cut away what was needed, so there can be two attempts.

    # An agent without execution: the query and nothing else. Handy to review queries
    # before anything runs, or when the application has its own access to the database.
    result = await query_only.run("Which actors appeared in The Matrix?", graph=GRAPH)
    show(result)
    print(f"  records: {result.records} (nothing was run)")
    # Q: Which actors appeared in The Matrix?
    # status: valid
    #   attempt 1 (filtered schema):
    #     MATCH (p:Person)-[:ACTED_IN]->(:Movie {title: 'The Matrix'}) RETURN p.name
    #   records: None (nothing was run)


async def together(agent: Text2Cypher) -> None:
    heading("6. Several questions at once with one agent")
    # An agent keeps nothing between questions, so it can be shared by concurrent tasks
    results = await asyncio.gather(
        agent.run("How many people were born in the United States?", graph=GRAPH),
        agent.run("Which studio produced the most movies?", graph=GRAPH),
        agent.run("Who directed Gravity?", graph=GRAPH),
    )
    for result in results:
        rows = result.records.rows if result.records else None
        print(f"  {result.question} -> {result.status}: {rows}")
    # How many people were born in the United States? -> valid: ((12,),)
    # Which studio produced the most movies? -> valid: (('Warner Bros.', 3),)
    # Who directed Gravity? -> valid: (('Alfonso Cuaron',),)


async def when_things_go_wrong(agent: Text2Cypher) -> None:
    heading("7. Errors")
    # Every error the library raises is a Text2CypherError. A graph that does not exist
    # fails before any call to the model is made.
    try:
        await agent.run("How many movies are there?", graph="no_such_graph")
    except GraphNotFoundError as error:
        print(f"  no such graph: {error}")
        # no such graph: The graph 'no_such_graph' does not exist, or has no labelled nodes or relationships

    # A failure in the middle of a run (the model's API down, a lost connection) is an
    # LLMError or a DatabaseError. They keep what was paid for before the failure.
    try:
        result = await agent.run("Who directed Cast Away?", graph=GRAPH)
        print(f"  {result.question} -> {result.status}")
        # Who directed Cast Away? -> valid   (valid, and no rows: nobody directed it here)
    except (LLMError, DatabaseError) as error:
        print(
            f"  stopped: {error}; attempts made: {len(error.attempts)}, tokens: {error.usage}"
        )
        # stopped: The LLM failed after 1 attempt(s) at 'Who directed Cast Away?': <the API's message>; attempts made: 1, tokens: Usage(...)


async def tour(
    client: FalkorDB,
    llm: AsyncOpenAI | AsyncChatClient,
    *,
    model: str | None = "gpt-4o",
    filter_model: str | None = "gpt-4o-mini",
) -> None:
    """Seed the graph, ask every kind of question, and delete the graph."""
    graph = client.select_graph(GRAPH)
    if GRAPH in await client.list_graphs():
        await graph.delete()
    await graph.query(SEED)
    try:
        agent = build_agent(client, llm, model=model, filter_model=filter_model)
        # No execution and no example values: the query is all there is to read
        query_only = Text2Cypher(
            client,
            llm,
            model=model,
            config=Config(execute=False, example_values=0),
        )
        await values(agent)
        await nodes(agent)
        await relationships_and_paths(agent)
        await nested(agent)
        await declined_and_query_only(agent, query_only)
        await together(agent)
        await when_things_go_wrong(agent)
    finally:
        await graph.delete()


async def main() -> None:
    client = FalkorDB(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", "6379")),
    )
    llm = AsyncOpenAI()  # reads OPENAI_API_KEY
    try:
        await tour(client, llm)
    finally:
        await client.aclose()
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
