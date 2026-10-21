# text2cypher

**Generate validated Cypher queries from natural language text with LLMs**
*Ask a question in plain language and get back a read-only Cypher query that has been checked against your [FalkorDB](https://www.falkordb.com/) graph.*

[![CI](https://github.com/Killeroid/text2cypher/actions/workflows/ci.yml/badge.svg)](https://github.com/Killeroid/text2cypher/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/text2cypher/badge/?version=latest)](https://text2cypher.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Killeroid/text2cypher/blob/main/LICENSE)

text2cypher asks an LLM to write the query, checks it against the real graph with [cypher-guardrail](https://github.com/Killeroid/cypher-guardrail), and sends what is wrong back to the LLM until the query passes or the attempts run out. A query that fails the checks is never handed back as if it were fine.

* **Grounded in your graph**: the prompt carries the schema read from the graph itself (labels, relationship types, properties with their value types, the paths that exist, and the indexes), cut down to what the question is about.
* **Validated before it is returned**: syntax, schema (labels, relationship types, paths and their direction) and property checks. A query that could write to the graph is never valid.
* **Self-correcting**: the problems found are written back to the LLM. If the cut-down schema does not get there, the loop moves to the full one.
* **Honest outcomes**: `valid`, `unanswerable` (the LLM declined) or `failed` (the corrections ran out), with every attempt, its issues, the schema it used and its token usage.
* **Example values**: the schema shows up to three values of each property, so the LLM writes values the way your graph does. They are real data sent to the LLM provider: `Config(example_values=0)` sends none.
* **Optional execution**: run the query read-only, with a timeout, and get the rows; a query that FalkorDB fails at run time is sent back to the LLM like any other problem.
* **Bring your own LLM**: a small protocol, `chat` (the conversation as turns) or just `complete`, with an OpenAI client included. Async throughout.

## Installation

Python 3.12 or later.

```sh
uv add text2cypher        # or: pip install text2cypher
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

Always look at `result.status`. `valid` means the query passed the checks, not that it means what you meant (see below).

There is also [a quickstart that needs no model and no server](https://github.com/Killeroid/text2cypher/blob/main/examples/quickstart.py): it starts an embedded FalkorDB and answers with a scripted LLM, including one correction.

## What is and is not checked

* **Checked**: the query is valid Cypher for FalkorDB, it cannot write, its parameters are provided (the prompt asks for values written in the query, so a `$name` is a problem), the labels, relationship types and paths exist in the direction used, and every property read exists on the label or relationship type it is read from.
* **Not checked**: that the query means what the question meant. A query can be valid and still answer another question, for example by using a property with a similar name. Look at the query, or at the rows, before you rely on them.
* **Known limits**: the prompts, the loop and the schema filter have been tried with Claude Haiku on two small test graphs (6 and 36 labels); the filter was measured offline on the larger one. How well they do on your data, on a large schema or with other models is not measured yet, so check the queries and the rows on your own questions (the [`evals/`](https://github.com/Killeroid/text2cypher/blob/main/evals/README.md) harness is there for that). Prompts and the filter's matching can change in a minor version; the public names (`text2cypher.__all__`, `text2cypher.llm` and `text2cypher.schema`) follow [semantic versioning](https://semver.org/).

## Documentation

The full documentation is at **[text2cypher.readthedocs.io](https://text2cypher.readthedocs.io/en/latest/)**:

* [Using text2cypher](https://text2cypher.readthedocs.io/en/latest/usage.html): the result, the configuration, running the query, errors
* [How it works](https://text2cypher.readthedocs.io/en/latest/how-it-works.html): the loop, the escalation and what it costs
* [The schema and its filter](https://text2cypher.readthedocs.io/en/latest/schema.html): what the LLM is shown, your own filter, keeping the schema between runs
* [LLM clients](https://text2cypher.readthedocs.io/en/latest/llm.html): your own client, other OpenAI-compatible models, `FakeLLM` for tests
* [What is and is not checked](https://text2cypher.readthedocs.io/en/latest/checks.html) and the [API reference](https://text2cypher.readthedocs.io/en/latest/reference.html)

For contributors: [how it is built, and why](https://github.com/Killeroid/text2cypher/blob/main/docs/architecture.md), and [a harness](https://github.com/Killeroid/text2cypher/blob/main/evals/README.md) that replays questions through the real pipeline with replies written by a model or a person.

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

Apache License 2.0: see [LICENSE](https://github.com/Killeroid/text2cypher/blob/main/LICENSE).
