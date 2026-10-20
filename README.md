# text2cypher

**Generate validated Cypher queries from natural language text with LLMs**
*Ask a question in plain language and get back a read-only Cypher query that has been checked against your [FalkorDB](https://www.falkordb.com/) graph.*

[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-brightgreen)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](CHANGELOG.md)

text2cypher asks an LLM to write the query, checks it against the real graph with [cypher-guardrail](https://github.com/Killeroid/cypher-guardrail), and sends what is wrong back to the LLM until the query passes or the attempts run out. A query that fails the checks is never handed back as if it were fine.

* **Grounded in your graph**: the prompt carries the schema read from the graph itself (labels, relationship types, properties with their value types, the paths that exist, and the indexes), cut down to what the question is about.
* **Validated before it is returned**: syntax, schema (labels, relationship types, paths and their direction) and property checks. A query that could write to the graph is never valid.
* **Self-correcting**: the problems found are written back to the LLM. If the cut-down schema does not get there, the loop moves to the full one.
* **Honest outcomes**: `valid`, `unanswerable` (the LLM declined) or `failed` (the corrections ran out), with every attempt, its issues, the schema it used and its token usage.
* **Optional execution**: run the final query read-only, with a timeout, and get the rows.
* **Bring your own LLM**: a one-method protocol, with an OpenAI client included. Async throughout.

## Installation

Python 3.12 or later. text2cypher is in early development and is not on PyPI yet; install it from a checkout:

```sh
uv add ./text2cypher        # or: pip install ./text2cypher
```

You need a FalkorDB graph that has data, reachable with the async client (`falkordb.asyncio`), and a model. The included client uses OpenAI's API and reads `OPENAI_API_KEY`.

## Quickstart

```python
import asyncio

from falkordb.asyncio import FalkorDB
from text2cypher import OpenAIClient, Text2Cypher


async def main() -> None:
    client = FalkorDB(host="localhost", port=6379)  # one client per event loop
    agent = Text2Cypher(client, OpenAIClient("your-model"))

    result = await agent.run("Which actors appeared in The Matrix?", graph="movies")
    print(result.status)  # "valid", "unanswerable" or "failed"
    print(
        result.query
    )  # MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name

    await client.aclose()


asyncio.run(main())
```

Always look at `result.status`. `valid` means the query passed the checks below, not that it means what you meant: see [What is and is not checked](#what-is-and-is-not-checked).

## How it works

```
question ─► read the schema ─► keep what the question needs ─► ask the LLM ─► validate ─┬─ valid ──────────► result
                                                                    ▲                    ├─ problems, attempts left ─► ask the LLM to correct ─┐
                                                                    │                    └─ attempts used up ─────────► failed                  │
                                                                    └───────────────── an empty answer: unanswerable ◄────────────────────────────┘
```

1. **Read the schema** from the graph: labels, relationship types, each property with the types of its values, and every one-hop path (`(:Person)-[:ACTED_IN]->(:Movie)`).
2. **Filter it** to what the question mentions (see [The schema and its filter](#the-schema-and-its-filter)).
3. **Ask the LLM** for one read-only query, in the filtered schema.
4. **Validate** the query against the graph with cypher-guardrail: syntax (nothing is run; FalkorDB is only asked to explain the query), then the labels, relationship types and paths it uses, and the properties it reads. All three always run, so one pass reports every kind of problem.
5. **Correct**: if there are problems, the LLM gets the question, its query and the problems in plain words (with the fixed path when only the direction was wrong), and answers again. The first `attempts_w_filtered` corrections use the filtered schema, the later ones the full schema.
6. **Stop** when a query passes (`valid`), when the corrections run out (`failed`), or when the LLM says the question cannot be answered with this graph by returning an empty answer (`unanswerable`). If it declines on the filtered schema, it is asked again once with the full schema before that counts.

The LLM sees your question, the schema (names and value types, no data) and its own earlier attempts with what was wrong with them. It never sees data from your graph.

## The result

```python
result = await agent.run("Which actors appeared in The Matrix?", graph="movies")

print(result.status)  # "valid"
print(result.query)  # the last query, empty if the LLM declined
print(
    result.usage
)  # Usage(input_tokens=310, output_tokens=24), or None if a call did not report it

for attempt in result.attempts:  # every query, in order
    print(
        attempt.schema_scope
    )  # "filtered" or "full": the schema this query was written for
    print(attempt.query)
    if attempt.validation:  # None when the query is empty
        print(attempt.validation.valid, attempt.validation.issues)
    print(attempt.usage)
```

| `status` | Meaning |
|---|---|
| `valid` | The last query passed every check. |
| `unanswerable` | The LLM declined: the question is not about this graph, or asks to change data. There is no query. |
| `failed` | The corrections ran out. The last attempt has the last query and what was still wrong. |

Each issue (`Issue`) has the validator it came from (`source`: `syntax`, `schema` or `properties`), a `code`, a `description`, and the structured fields of cypher-guardrail's issue in `raw` (`label`, `suggested_path`, ...), so nothing has to be read out of text.

## Configuration

```python
from text2cypher import Config

agent = Text2Cypher(client, llm, config=Config(attempts_w_filtered=2, total_attempts=4))
```

| Field | Default | Meaning |
|---|---|---|
| `attempts_w_filtered` | `3` | Corrections made with the filtered schema. |
| `total_attempts` | `5` | Corrections in all, filtered and full. The first query is not a correction, so the LLM is asked at most `total_attempts + 1` times (plus one retry on the full schema if it declines on the filtered one). |
| `execute` | `False` | Run the final query, see below. |
| `execute_timeout_ms` | `10000` | Stop an execution after this many milliseconds; `None` for no limit. |
| `schema_refresh_seconds` | `None` | Keep a graph's schema and check it for changes at most this often; `None` reads it on every run (see [Keeping the schema](#keeping-the-schema)). |

`Config` rejects negative values and `total_attempts < attempts_w_filtered`.

## Running the query

With `Config(execute=True)` a `valid` query is also run, read-only (a write is rejected by FalkorDB itself), and the rows are on the result:

```python
agent = Text2Cypher(client, llm, config=Config(execute=True))
result = await agent.run("Which actors appeared in The Matrix?", graph="movies")

print(result.records.columns)  # ('p.name',)
print(result.records.rows)  # (('Keanu Reeves',),)
print(result.execution_error)  # None
```

If FalkorDB rejects or fails the query, it runs past the timeout, or it cannot be reached, the message is in `result.execution_error` and the status stays `valid` (the query did pass the checks), so the LLM calls you paid for are not lost: run `result.query` yourself or try again. Values in the rows are the client's own objects (a node stays a `falkordb` `Node`).

## The schema and its filter

The schema is read from the graph on every `run` (or kept, see [Keeping the schema](#keeping-the-schema)). By default it is filtered with `filter_schema`: a label or relationship type is kept when a word of its name has the same stem as a word of the question (`acted` finds `ACTED_IN`, `movies` finds `Movie`; names are split at underscores and camelCase), together with the paths it is part of and the labels and types on those paths, with their properties. A question that matches nothing keeps the whole schema. The filter is lexical: it does not know that "people" means `Person`, so it falls back to the full schema rather than guess.

Pass your own filter, any function from the full schema and the question to a smaller schema, or `None` to always use the full schema:

```python
from text2cypher import GraphSchema


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

A filter that returns the whole schema counts as no filter.

### Keeping the schema

Reading the schema is a full scan of the graph, because the types of the values need one. With `schema_refresh_seconds` the schema is kept, and a run that comes after that many seconds first checks whether the graph changed (a few cheap queries) and reads the schema again only if it did:

```python
agent = Text2Cypher(client, llm, config=Config(schema_refresh_seconds=300))
```

`0` checks on every run. The check is a heuristic (removing one relationship and adding another leaves the counts as they were), and a change between checks is not in the prompt; a query that uses a new label or property is still accepted, because the validators ask the database before they report a problem. Either way the validators answer from the same snapshot the prompt was written from, so a query that is fine costs FalkorDB one request, to plan it.

## Using another LLM

An LLM is anything with one method: it gets a system message and a user message, and returns the text and, if it knows them, the tokens used.

```python
from text2cypher import Completion, Text2Cypher, Usage


class MyClient:
    async def complete(self, system: str, user: str) -> Completion:
        text, tokens_in, tokens_out = await ask_my_model(system, user)
        return Completion(text, Usage(input_tokens=tokens_in, output_tokens=tokens_out))


agent = Text2Cypher(client, MyClient())
```

The reply should hold the query in one fenced block (` ```cypher ... ``` `); an empty block means "cannot be answered". Text around the block is ignored.

`OpenAIClient(model, *, client=None, temperature=0, system_role=True)` uses the Chat Completions API. Pass your own `AsyncOpenAI` for another endpoint, for example an OpenAI-compatible server, `temperature=None` for a model that only accepts its own, and `system_role=False` for a model that has no system role (one user message is sent):

```python
from openai import AsyncOpenAI
from text2cypher import OpenAIClient

llm = OpenAIClient(
    "a-local-model",
    client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
)
```

## Testing your code

`FakeLLM` replays scripted replies and records what it was asked, so you can test code that uses text2cypher without a model:

```python
from text2cypher import FakeLLM, Text2Cypher

llm = FakeLLM(["```cypher\nMATCH (p:Person) RETURN p.name\n```"])
result = await Text2Cypher(client, llm).run("Who is in the graph?", graph="movies")

assert result.status == "valid"
assert llm.calls[0].user == "Question: Who is in the graph?"
```

## Errors

* `GraphNotFoundError` (a `ValueError`): the graph does not exist, or has no labelled nodes or relationships. Raised by `run` before any LLM call, so a typo in the graph name costs nothing.
* Errors from your LLM client and from the database while reading the schema or validating are not caught: they reach you as they are.
* Errors while executing the final query are recorded in `result.execution_error`, as above.

## What is and is not checked

* **Checked**: the query is valid Cypher for FalkorDB, it cannot write, its parameters are provided (the prompt asks for values written in the query, so a `$name` is a problem), the labels, relationship types and paths exist in the direction used, and every property read exists on the label or relationship type it is read from.
* **Not checked**: that the query means what the question meant. A query can be valid and still answer another question, for example by using a property with a similar name. Look at the query, or at the rows, before you rely on them.
* **Early days**: the prompts and the loop have been tried with one small model on a small graph, and the filter has not been measured on a large schema. Both are likely to need work.
* **Cost**: the schema is read from the graph on every `run` unless you set `schema_refresh_seconds`; a validation then costs one request to FalkorDB when the query is fine.
* **One event loop**: the FalkorDB async client is bound to the loop that first used it, so use one `Text2Cypher` per loop.

## Documentation

* [`docs/architecture.md`](docs/architecture.md): how it is built, and why
* [`evals/`](evals/README.md): a harness that replays questions through the real pipeline with replies written by a model or a person
* [CHANGELOG](CHANGELOG.md)

## Development

```sh
uv sync                              # create the environment (Python >= 3.12 also installs FalkorDBLite for the tests)
uv run pytest                        # the tests, on an embedded FalkorDB
uv run pytest -m "not db"            # only the tests that need no database (under ten seconds)
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run lint-imports
uv run coverage run -m pytest && uv run coverage report
```

Contributions are welcome: please open an issue or a pull request.

## License

Apache License 2.0: see [LICENSE](LICENSE).
