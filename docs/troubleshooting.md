# When a question fails

A run always ends in a result or an error that says what happened. This page is what to look at, and what to try, for each way a question can go wrong.

## See what happened

Every attempt is kept on the result, with the query, how it was validated, and what it cost:

```python
result = await agent.run("Which actors appeared in The Matrix?", graph="movies")

print(result.status)  # "valid", "unanswerable" or "failed"
for number, attempt in enumerate(result.attempts, start=1):
    print(f"{number}. [{attempt.schema_scope}] {attempt.query or '(declined)'}")
    if attempt.auto_fixed:
        print("   repaired without the LLM")
    if attempt.execution_error:
        print(f"   FalkorDB: {attempt.execution_error}")
    if attempt.validation:
        for issue in attempt.validation.issues:
            print(f"   {issue.code}: {issue.description}")
```

`attempt.schema_scope` says whether the attempt was written with the filtered or the full schema (see [](schema.md)), `attempt.usage` has its tokens, and a query that was repaired without the LLM follows the one the LLM wrote with `auto_fixed` set.

To see what the LLM was shown, print the schema (see [](schema.md#the-schema)). It is what a question can use: a label, a property or a relationship that is not in it cannot be asked about.

## By what you see

| What you see | What it means | What to try |
|---|---|---|
| Status `unanswerable` | The LLM declined, also with the full schema: it sees no way to answer from the graph, or the question asks to change data or is not a request for a query. | Check that the graph has the labels, relationships and properties the question needs (the schema above). Ask for something the graph holds. |
| Status `failed` | The corrections ran out. The last attempt has the last query and its problems. | Read the problems of each attempt: a name that is not in the graph points to a question the schema cannot answer, or a filter that kept too little (`schema_filter=None` shows the model the whole schema). Raise `attempts_w_filtered` and `total_attempts`, try a stronger model, or turn on `autofix_typos` for misspelled names. |
| Status `valid`, rows empty | The query is correct and nothing matches. An empty result is not an error, and it is not corrected. | A value spelled differently than in the graph is the usual cause: `example_values` shows the model how values are written (it is on by default; see [](schema.md#example-values)). Check the value in the question exists in the data. |
| Status `valid`, rows that look wrong | The checks say the query is valid against the graph, not that it answers the question. | Read `result.query`. This is the limit of validation (see [](checks.md)); the evaluation harness in `evals/` is for measuring it on your own questions. |
| Status `valid`, `execution_error` says "timed out" | The query is valid and costs more than `execute_timeout_ms`. | Raise `execute_timeout_ms`, or ask a narrower question. `check_performance=True` reports queries that scan every node or have no upper bound. |
| `execution_error` on a `failed` result | FalkorDB rejected or failed the last queries when they ran (a type error on a value, a division by zero), and the corrections ran out. | The error of each attempt is on `attempt.execution_error`. The model is told it after each failure, as the next turn. |
| An [`LLMError`](usage.md#errors) | Your LLM client raised: a rate limit, a lost connection, a bad key, or a call that took longer than the timeout. | The cause is `error.__cause__`. The attempts made before it are on `error.attempts`, with `error.usage`. For `OpenAIClient`, see `timeout` and `max_retries` in [](llm.md). |
| A [`DatabaseError`](usage.md#errors) | FalkorDB failed while a query was being validated. | Check the connection to the server. The attempts so far, and the query being validated, are on the error. |
| [`GraphNotFoundError`](usage.md#errors) | The graph does not exist, or has no labelled nodes or relationships. | Check the name of the graph. FalkorDB has no graph without data. |
| The prompt does not show a change you made to the graph | The schema is kept (`schema_refresh_seconds`), or its example values are, because they are read once for a structure of the graph. | Call `agent.clear_schema(graph)` (or `agent.clear_schema()` for every graph). The next run reads it again. A query that uses a new name is still validated against the graph. |

## The cost of trying again

A question costs one LLM call when the first query passes, and at most `total_attempts + 1` calls (plus one when an empty answer on the filtered schema is retried on the full one). A query repaired without the LLM costs none. See [](how-it-works.md#cost).
