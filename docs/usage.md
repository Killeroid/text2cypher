# Using text2cypher

## Installation

Python 3.12 or later.

```sh
uv add text2cypher        # or: pip install text2cypher
```

You need a FalkorDB graph that has data, reachable with the async client (`falkordb.asyncio`), and a model. The included client uses OpenAI's API and reads `OPENAI_API_KEY`; see [LLM clients](llm.md) for other models.

## Asking a question

```python
import asyncio

from falkordb.asyncio import FalkorDB
from openai import AsyncOpenAI
from text2cypher import Text2Cypher


async def main() -> None:
    client = FalkorDB(host="localhost", port=6379)  # one client per event loop
    agent = Text2Cypher(client, AsyncOpenAI(), model="your-model")

    result = await agent.run("Which actors appeared in The Matrix?", graph="movies")
    print(result.status)  # "valid", "unanswerable" or "failed"
    print(result.query)

    await client.aclose()


asyncio.run(main())
```

`Text2Cypher` takes the FalkorDB client and the LLM. `run` takes the question and the name of the graph, and reads the graph's schema every time it is called. The client is bound to the event loop that first uses it, so use one `Text2Cypher` per loop. Always look at `result.status` before you use `result.query`.

## The result

`run` returns a {py:class}`~text2cypher.Result`:

```python
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
| `failed` | The corrections ran out. The last attempt has the last query and what was still wrong. It can also have no query at all: the LLM sent no text twice in a row (`attempt.empty_reply`). |

Each {py:class}`~text2cypher.Issue` in `attempt.validation.issues` says which validator found it (`source`: `syntax`, `schema` or `properties`), has a `code` and a `description`, and keeps cypher-guardrail's own structured issue in `raw`, so nothing has to be read out of text. See [What is and is not checked](checks.md).

`result.usage` adds up the tokens of all attempts, and is `None` if any call did not report them.

## Configuration

```python
from text2cypher import Config

agent = Text2Cypher(client, llm, config=Config(attempts_w_filtered=2, total_attempts=4))
```

| Field | Default | Meaning |
|---|---|---|
| `attempts_w_filtered` | `5` | Corrections made with the filtered schema. |
| `total_attempts` | `10` | Corrections in all, filtered and full. The first query is not a correction, so the LLM is asked at most `total_attempts + 1` times (plus one retry on the full schema if it declines on the filtered one). |
| `execute` | `False` | Run a query that passed validation, and correct it if FalkorDB fails it; see below. |
| `execute_timeout_ms` | `10000` | Stop an execution after this many milliseconds; `None` for no limit. |
| `check_types` | `True` | Report a property compared with a value of another type, and `sum`, `avg` and the like of a property that holds no numbers. See [What is and is not checked](checks.md). |
| `check_performance` | `False` | Report a cartesian product, a scan of every node and a path with no upper bound. See [What is and is not checked](checks.md). |
| `use_parameters` | `False` | The LLM puts the values in a `CYPHER` header and uses parameters, which FalkorDB can plan once and reuse. `Result.query` is still one string that runs as it is. A header with commas or without `CYPHER` is repaired without the LLM. |
| `example_values` | `3` | Example values of each property shown to the LLM; they are data from your graph that goes to the LLM provider, so `0` sends none. See [Example values](schema.md#example-values). |
| `use_skills` | `False` | Offer the LLM short guides to parts of FalkorDB's Cypher that are easy to get wrong, to read when a question needs them. Each read is another call to the model. To use your own, see [Your own skills](how-it-works.md#your-own-skills). A smaller model benefits from turning this on; a stronger one gains little (see [Skills](how-it-works.md#skills)). Check it on your own questions. |
| `autofix_typos` | `False` | Fix a misspelled label, relationship type or property without the LLM when exactly one name of the graph is a likely typo of it; the fixed query is validated again. See [How it works](how-it-works.md). |
| `schema_refresh_seconds` | `None` | Keep a graph's schema and check it for changes at most this often; `None` reads it on every run. See [The schema and its filter](schema.md). |

`Config` rejects negative values and `total_attempts < attempts_w_filtered`. The LLM is sent your question, the schema, up to `example_values` real values of each property, and the problems of its earlier attempts, and, with `execute`, FalkorDB's error for a query that fails.

### Model Selection

Choose LLM models based on your requirements. Different models provide different balances of speed and accuracy and cost. See [LLM clients](llm.md) for more details.


### Schema filtering

On a large graph the whole schema makes a long prompt and many ways to go wrong, so by default the schema is cut down to what the question mentions. The default is filtering by word matching through a lexical filter. The lexical filter does not know that "people" means `Person` or that "films" means `Movie`, and when nothing matches it keeps the whole schema. On a large graph that is the expensive case.

You can switch to an LLM based filter which asks a model which labels and relationship types the question needs instead:

```python
from text2cypher import LLMSchemaFilter, OpenAIClient, Text2Cypher

# the main model chooses too
agent = Text2Cypher(client, llm, schema_filter=LLMSchemaFilter())
```

See [The schema and its filter](schema.md).



## Running the query

With `Config(execute=True)` a query that passed validation is also run, read-only (FalkorDB itself rejects a write), and the rows are on the result:

```python
agent = Text2Cypher(client, llm, config=Config(execute=True))
result = await agent.run("Which actors appeared in The Matrix?", graph="movies")

print(result.records.columns)  # ('p.name',)
print(result.records.rows)  # (('Keanu Reeves',), ('Carrie-Anne Moss',))
print(result.execution_error)  # None
```

If FalkorDB rejects or fails the query (a type error on a value, a division by zero, a bad argument to a function: what the validators cannot see before it runs), the error is sent back to the LLM for a correction, as the next turn of the conversation, and the corrected query is validated and run in turn. This costs a correction like any other, so it uses the same `total_attempts`. If the corrections run out on queries that FalkorDB fails, the status is `failed`, `result.query` is the last query, and `result.execution_error` is the last error (each attempt has its own in `attempt.execution_error`).

If the query runs past the timeout, or FalkorDB cannot be reached, the message is in `result.execution_error` and the status stays `valid`: the query did pass the checks, that says nothing about the query, and the LLM calls you paid for are not lost. Run `result.query` yourself or try again. Queries that are `unanswerable` are never run. Values in the rows are the client's own objects (a node stays a `falkordb` `Node`). An error message from FalkorDB can name a value from your data, and it is sent to the LLM; the rows are not.

## Errors

Every error the library raises itself is a {py:class}`~text2cypher.Text2CypherError`, so one `except` catches them all.

* {py:class}`~text2cypher.GraphNotFoundError` (also a `ValueError`): the graph does not exist, or has no labelled nodes or relationships. Raised by `run` before any LLM call, so a typo in the graph name costs nothing.
* {py:class}`~text2cypher.LLMError`: your LLM client raised while a question was being answered (a rate limit, a lost connection, a bad key; any exception it raises). The attempts made before it were paid for, so they are kept on the error: `error.attempts` (each with its validation, as on a result), `error.usage` (their tokens, or `None` if a client did not report them; the call that failed is not counted, because nothing says what it cost) and `error.question`. The exception your client raised is `error.__cause__`. Cancelling the task is not an error and passes through.
* {py:class}`~text2cypher.DatabaseError`: FalkorDB failed (a lost connection, a failing server) while a query was being validated. The same fields; the last attempt is the query that was being validated, with `validation=None`, because it was written and paid for.

```python
from text2cypher import LLMError

try:
    result = await agent.run("Who acted in Heat?", graph="movies")
except LLMError as error:
    print(f"The model failed after {len(error.attempts)} attempts: {error.__cause__!r}")
    print(error.usage)  # what the earlier attempts cost
```

A catch of your provider's own exception (for example the OpenAI `RateLimitError`) around `run` now sees an `LLMError` instead, with that exception as `__cause__`.

* Errors from the database while reading the schema, before any LLM call, are not caught: they reach you as they are.
* A timeout or a lost connection while executing is recorded in `result.execution_error`, as above; an error from FalkorDB about the query is corrected, as above.

If a question did not go as you wanted, [When a question fails](troubleshooting.md) says what to look at for each outcome.

## Try it without a server or a model

This example starts an embedded FalkorDB ([FalkorDBLite](https://pypi.org/project/falkordblite/)) with a small film graph and answers with a scripted LLM, so the whole flow, including a correction, runs in a few seconds. It is in the repository as `examples/quickstart.py`.

```{literalinclude} ../examples/quickstart.py
:language: python
:start-at: import asyncio
```

It prints:

```text
status: valid
attempt 1: MATCH (m:Movie {title: 'The Matrix'})-[:ACTED_IN]->(p:Person) RETURN p.name
  problem: The specified path (:Movie)-[:ACTED_IN]->(:Person) has the opposite direction in the knowledge graph schema. The path (m:Movie {title: 'The Matrix'})-[:ACTED_IN]->(p:Person) in the Cypher statement should be changed to  (m:Movie {title: 'The Matrix'})<-[:ACTED_IN]-(p:Person).
attempt 2: MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name
columns: ('p.name',)
rows: (('Keanu Reeves',), ('Carrie-Anne Moss',))
```
