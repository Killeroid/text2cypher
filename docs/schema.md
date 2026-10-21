# The schema and its filter

## The schema

FalkorDB has no declared schema, so text2cypher describes the graph by what it contains. On every `run` (see [Keeping the schema](#keeping-the-schema) to change that) it captures a [cypher-guardrail](https://github.com/Killeroid/cypher-guardrail) `Snapshot` of the graph, with the types of the values, and builds a {py:class}`~text2cypher.GraphSchema` from it:

* `node_properties`: for each label, its properties, each with the types of its values (`{"year": ("Integer",)}`; a property with mixed values lists them all);
* `relationship_properties`: the same for each relationship type;
* `paths`: the one-hop paths that exist, as `Path(source, type, target)`. A node with several labels gives one path per label.
* `functions`: the user-defined functions of the FalkorDB instance (`library.function`), listed under `User-defined functions:` in the prompt with the rule to call only those listed. They belong to the instance, so the filter keeps all of them.
* `indexes`: the indexes on properties, as `Index(entity, property, type)` with `type` one of `RANGE`, `FULLTEXT` or `VECTOR`. They are listed under `Indexes:` in the prompt, and the prompt gets a rule for each kind that exists: compare an indexed property directly (`=`, `<`, `>`, `IN`) because `<>`, `NOT`, `STARTS WITH`, `CONTAINS` and a function on the property cannot use the index (checked on FalkorDB); search words with `db.idx.fulltext.queryNodes` only when the question is a word search; and never call `db.idx.vector.*`, because there is no embedding for the question.
* `node_examples` and `relationship_examples`: up to `Config.example_values` (3 by default) example values of each plain property, written as in a query, which are shown after the type. See [Example values](#example-values).

Labels, relationship types and properties whose name contains a backtick are left out: FalkorDB cannot parse an escaped backtick, so no query can use them.

You can do the same by hand, for example to look at what the LLM would get:

```python
from cypher_guardrail import Snapshot
from text2cypher.schema import render_schema, schema_from_snapshot

snapshot = await Snapshot.capture_async(client, "movies", types=True)
schema = schema_from_snapshot(snapshot)
print(render_schema(schema))
```

```text
Node properties:
Genre {name: STRING}
Movie {rating: FLOAT, title: STRING, year: INTEGER}
Person {born: INTEGER, name: STRING}
Relationship properties:
ACTED_IN {role: STRING}
The relationships:
(:Movie)-[:IN_GENRE]->(:Genre)
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:KNOWS]->(:Person)
```

This text is what goes into the prompt. Names that are not plain identifiers are written in backticks, and a section with nothing in it is left out.

## Example values

`Config(example_values=3)` is the number of values shown for each property of a plain type (text, number, boolean; not a list, map, point or vector), and `0` shows none:

```text
Node properties:
Movie {rating: FLOAT (e.g. 8.7), title: STRING (e.g. 'Matrix'), year: INTEGER (e.g. 1999)}
```

They tell the LLM how a value is written in your graph: `'Sales'` or `'sales'`, `1999` or `'1999'`. When the schema has any, the prompt adds a rule to write a value the way the examples are written.

* **They are data from your graph, and they go to the LLM provider.** Turn them off with `example_values=0` if that is not acceptable. Nothing is read from the graph for them then.
* Each property costs one small read-only query that stops after a few rows, with a 2 second timeout; a property that fails or is slow gets none. They are read once for a structure of the graph and kept with the schema, so a question does not pay for them again until a label, relationship type, property or index appears or goes. A value that changes inside the same structure is not noticed.
* A long string is cut at 40 characters, and the filter below keeps the examples of what it keeps.

## Filtering

On a large graph the whole schema makes a long prompt and many ways to go wrong, so by default the schema is cut down to what the question mentions with {py:func}`~text2cypher.filter_schema`:

* A label or relationship type is kept when a word of its name has the same stem as a word of the question: `acted` finds `ACTED_IN`, and `movies` finds `Movie`. Names are split at underscores, spaces and symbols, at camelCase boundaries (`MovieGenre`), after an acronym (`HTTPServer`) and between letters and digits. Stopwords such as `in` are left out of a name, so a question's "in" does not find `IN_GENRE`, unless a name is made only of stopwords.
* A match keeps the paths it is part of, and the labels and relationship types on those paths with all their properties, because the LLM needs them to write the query.
* A question that matches nothing keeps the whole schema.

For "Which genre is the Matrix in?" on the graph above, the LLM gets:

```text
Node properties:
Genre {name: STRING}
Movie {rating: FLOAT, title: STRING, year: INTEGER}
The relationships:
(:Movie)-[:IN_GENRE]->(:Genre)
```

The filter is lexical. It does not know that "people" means `Person` or that "films" means `Movie`, and it does not match property names, so for such a question it keeps the whole schema (or too little). Too little is not final: the first `attempts_w_filtered` corrections use the filtered schema, the later ones the full schema, and an empty answer on the filtered schema is retried on the full one (see [](how-it-works.md)).

## Your own filter

The filter keeps the indexes of what it keeps. A filter is any function from the full schema and the question to a smaller schema ({py:data}`~text2cypher.schema._filter.SchemaFilter`). Pass it as `schema_filter`, or pass `None` to always use the full schema:

```python
from text2cypher import GraphSchema, Text2Cypher


def keep_people_and_movies(schema: GraphSchema, question: str) -> GraphSchema:
    keep = {"Person", "Movie"}
    return GraphSchema(
        node_properties={k: v for k, v in schema.node_properties.items() if k in keep},
        relationship_properties=schema.relationship_properties,
        paths=tuple(p for p in schema.paths if p.source in keep and p.target in keep),
        indexes=tuple(i for i in schema.indexes if i.entity in keep),
    )


agent = Text2Cypher(
    client, llm, schema_filter=keep_people_and_movies
)  # or schema_filter=None
```

A filter that returns the whole schema counts as no filter, so no attempt is labelled "filtered" and nothing is retried. The filter is synchronous.

## Keeping the schema

By default the schema is read from the graph on every `run`. That is a full scan, because the types of the values need one. With `Config(schema_refresh_seconds=...)` the snapshot is kept instead, and a run that comes after that many seconds first checks whether the graph changed (a few cheap queries: the names, the indexes and the number of nodes and relationships) and reads it again only if it did:

```python
from text2cypher import Config

agent = Text2Cypher(client, llm, config=Config(schema_refresh_seconds=300))
```

`0` checks on every run. The check is a heuristic: removing one relationship and adding another leaves the counts as they were. Between checks a change to the graph is not in the prompt, but the validators ask the database before they report a problem, so a query that uses a new label or property is still accepted.

To forget what is kept, call `agent.clear_schema("movies")` for one graph or `agent.clear_schema()` for all of them. The next run reads the schema again, and the example values too, which are kept while the structure of the graph is the same and so do not notice a changed value. It queries nothing by itself.

In both cases the validators answer from the snapshot the prompt was written from. They ask FalkorDB only to plan the query and to confirm a problem before reporting it, so a query that is fine costs one request.
