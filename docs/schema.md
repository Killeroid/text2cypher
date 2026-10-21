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

The filter is lexical. It does not know that "people" means `Person` or that "films" means `Movie`, and it does not match property names, so for such a question it keeps the whole schema (or too little). To let a model choose, see [Asking an LLM to filter](#asking-an-llm-to-filter). Too little is not final: the first `attempts_w_filtered` corrections use the filtered schema, the later ones the full schema, and an empty answer on the filtered schema is retried on the full one (see [How it works](how-it-works.md)).

## Your own filter

A filter is any function from the full schema and the question to a smaller schema ({py:data}`~text2cypher.schema._filter.SchemaFilter`). Pass it as `schema_filter`, or pass `None` to always use the full schema:

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

A filter that returns the whole schema counts as no filter, so no attempt is labelled "filtered" and nothing is retried.

To keep a label, a relationship type and what goes with them (the paths they are on, those paths' ends, and the properties, indexes and example values of all of them), end with {py:func}`~text2cypher.schema.restrict_schema`:

```python
from text2cypher.schema import restrict_schema


def keep_people(schema: GraphSchema, question: str) -> GraphSchema:
    return restrict_schema(schema, ["Person"], ["KNOWS"])
```

A filter may also be asynchronous, and it may ask a model: see below.

## Asking an LLM to filter

The lexical filter does not know that "people" means `Person` or that "films" means `Movie`, and when nothing matches it keeps the whole schema. On a large graph that is the expensive case. {py:class}`~text2cypher.LLMSchemaFilter` asks a model which labels and relationship types the question needs instead:

```python
from text2cypher import LLMSchemaFilter, OpenAIClient, Text2Cypher

# the main model chooses too
agent = Text2Cypher(client, llm, schema_filter=LLMSchemaFilter())

# or a cheaper model, for the choosing only
agent = Text2Cypher(
    client, llm, schema_filter=LLMSchemaFilter(OpenAIClient("a-small-model"))
)

# with the OpenAI SDK's client, the same client asks another model
agent = Text2Cypher(
    client,
    AsyncOpenAI(),
    model="your-model",
    schema_filter=LLMSchemaFilter(model="a-small-model"),
)
```

* **Does it help?** On a 36-label schema (80 names) with 20 questions that use other words than the schema, the model chose the right part on the first try for 20 of 20 (Haiku) and 19 of 20 (Sonnet), where the word-matching filter did for 12 and 11. Every question still ended valid with either, because a miss is retried on the full schema: the LLM filter saves those retries and spends one call on every question, so the calls come out about even. Our recommendation is to use it on a large schema, and to leave the default for a small one.
* **A cheaper model.** `LLMSchemaFilter(OpenAIClient("a-small-model"))`, or an `AsyncOpenAI` with `model=`, uses a client of its own. With no client, `LLMSchemaFilter(model="a-small-model")` asks the main client with that model instead, the same SDK client and settings; that needs an OpenAI client as the main one, and says so if it is not.
* **It replaces the lexical filter.** Choosing it means it runs alone: the offline filter does not run first, and the model is shown the whole schema.
* **What the model sees:** the question, each label with its property names, and the paths between them. No types, no example values, no indexes. It answers with the labels and relationship types it needs, and those and the paths they are on are kept (the same as `restrict_schema`). A name that is not in the schema is dropped.
* **If it cannot be used, you lose nothing.** A reply that is not valid, names nothing in the schema or chooses nothing, and a call that fails, all keep the whole schema, so the question is written with everything. As with any filtered schema, one that turns out to be too small costs corrections, not the answer: the loop moves to the full schema after `attempts_w_filtered`.
* **Small schemas are not filtered.** Below `min_entities` labels plus relationship types (20 by default) the whole schema is kept without asking, because there is little to cut. `min_entities=0` always asks.
* **It costs a call.** The filter's tokens are on `result.filter_usage` and in `result.usage`, and in the `usage` of an `LLMError`. The default filter asks no model, so its `filter_usage` is zero.

A filter you write may be `async` too. It returns a `GraphSchema`, or a {py:class}`~text2cypher.FilterResult` that also says what it cost, which is how it reports tokens ({py:class}`~text2cypher.AsyncSchemaFilter`).

## Keeping the schema

By default the schema is read from the graph on every `run`. That is a full scan, because the types of the values need one. With `Config(schema_refresh_seconds=...)` the snapshot is kept instead, and a run that comes after that many seconds first checks whether the graph changed (a few cheap queries: the names, the indexes and the number of nodes and relationships) and reads it again only if it did:

```python
from text2cypher import Config

agent = Text2Cypher(client, llm, config=Config(schema_refresh_seconds=300))
```

`0` checks on every run. The check is a heuristic: removing one relationship and adding another leaves the counts as they were. Between checks a change to the graph is not in the prompt, but the validators ask the database before they report a problem, so a query that uses a new label or property is still accepted.

To forget what is kept, call `agent.clear_schema("movies")` for one graph or `agent.clear_schema()` for all of them. The next run reads the schema again, and the example values too, which are kept while the structure of the graph is the same and so do not notice a changed value. It queries nothing by itself.

In both cases the validators answer from the snapshot the prompt was written from. They ask FalkorDB only to plan the query and to confirm a problem before reporting it, so a query that is fine costs one request.
