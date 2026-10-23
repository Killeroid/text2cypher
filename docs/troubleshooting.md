# When a question fails

A run always ends in a result or an error that says what happened. This page is what to look at, and what to try, for each way a question can go wrong.

## See what happened

Every attempt is kept on the result, with the query, how it was validated, and what it cost:

```python
result = await agent.run("Which actors appeared in The Matrix?", graph="movies")

print(result.status)  # "valid", "unanswerable" or "failed"
for number, attempt in enumerate(result.attempts, start=1):
    shown = attempt.query or ("(no reply)" if attempt.empty_reply else "(declined)")
    print(f"{number}. [{attempt.schema_scope}] {shown}")
    if attempt.finish_reason not in (None, "stop"):
        print(f"   the reply ended: {attempt.finish_reason}")
    if attempt.skills:
        print(f"   read: {', '.join(attempt.skills)}")
    if attempt.auto_fixed:
        print("   repaired without the LLM")
    if attempt.execution_error:
        print(f"   FalkorDB: {attempt.execution_error}")
    if attempt.validation:
        for issue in attempt.validation.issues:
            print(f"   {issue.code}: {issue.description}")
```

`attempt.schema_scope` says whether the attempt was written with the filtered or the full schema (see [The schema and its filter](schema.md)), `attempt.usage` has its tokens, and a query that was repaired without the LLM follows the one the LLM wrote with `auto_fixed` set.

To see what the LLM was shown, print the schema (see [The schema](schema.md#the-schema)). It is what a question can use: a label, a property or a relationship that is not in it cannot be asked about.

## By what you see

| What you see | What it means | What to try |
|---|---|---|
| Status `unanswerable` | The LLM declined (an empty block, or a `refusal` from the model), also with the full schema: it sees no way to answer from the graph, or the question asks to change data or is not a request for a query. | Check that the graph has the labels, relationships and properties the question needs (the schema above). Ask for something the graph holds. |
| Status `failed` with no query | The LLM sent no text. That is not a decline: the attempts have `empty_reply` set, and the snippet above shows them as "(no reply)". `attempt.finish_reason` says why when the client knows. `length`: the reply hit a token limit, so raise it (a model that thinks first can use it all before it writes anything). `content_filter`: the provider blocked it. Neither is asked again. Anything else was asked again once, and came back empty twice. | Read `finish_reason`. If it is `length`, raise the limit. If it is `content_filter`, change the question or the provider's settings. Otherwise try again, and look at the model or the provider. |
| Status `failed` | The corrections ran out. The last attempt has the last query and its problems. | Read the problems of each attempt: a name that is not in the graph points to a question the schema cannot answer, or a filter that kept too little (`schema_filter=None` shows the model the whole schema, and an [`LLMSchemaFilter`](schema.md#asking-an-llm-to-filter) lets a model choose). Raise `attempts_w_filtered` and `total_attempts`, try a stronger model, or turn on `autofix_typos` for misspelled names. |
| Status `valid`, rows empty | The query is correct and nothing matches. An empty result is not an error, and it is not corrected. | A value spelled differently than in the graph is the usual cause: `example_values` shows the model how values are written (it is on by default; see [Example values](schema.md#example-values)). Check the value in the question exists in the data. |
| Status `valid`, rows that look wrong | The checks say the query is valid against the graph, not that it answers the question. | Read `result.query`. This is the limit of validation (see [What is and is not checked](checks.md)); the evaluation harness in `evals/` is for measuring it on your own questions. |
| Status `valid`, `execution_error` says "timed out" | The query is valid and costs more than `execute_timeout_ms`. | Raise `execute_timeout_ms`, or ask a narrower question. `check_performance=True` reports queries that scan every node or have no upper bound. |
| `execution_error` on a `failed` result | FalkorDB rejected or failed the last queries when they ran (a type error on a value, a division by zero), and the corrections ran out. | The error of each attempt is on `attempt.execution_error`. The model is told it after each failure, as the next turn. |
| An [`LLMError`](usage.md#errors) | Your LLM client raised: a rate limit, a lost connection, a bad key, or a call that took longer than the timeout. | The cause is `error.__cause__`. The attempts made before it are on `error.attempts`, with `error.usage`. For the OpenAI SDK's client, see `timeout` and `max_retries` in [Model options](llm.md#model-options). |
| A [`DatabaseError`](usage.md#errors) | FalkorDB failed while a query was being validated. | Check the connection to the server. The attempts so far, and the query being validated, are on the error. |
| [`GraphNotFoundError`](usage.md#errors) | The graph does not exist, or has no labelled nodes or relationships. | Check the name of the graph. FalkorDB has no graph without data. |
| The LLM schema filter seems to do nothing, and every attempt used the full schema | A filter never ends a question: when its call fails, or its reply cannot be used, it keeps the whole schema, and the failure is not reported. A model that refuses a setting is the usual cause, for example `temperature` or `reasoning_effort` for a reasoning model. | Look at `result.filter_usage`: zero means no model was asked or the call failed. Ask the filter's model with the settings it takes, in `model_options` (see [LLM clients](llm.md#model-options)), and try the client of the filter on its own. |
| The data dictionary is not in the prompt, or is missing a name | A dictionary with no text for the graph shows nothing, a name the graph does not have is left out, and a blank description is shown as no description. | `dictionary.check(schema)` says which (see [Describing the graph](schema.md#describing-the-graph)). |
| The prompt does not show a change you made to the graph | The schema is kept (`schema_refresh_seconds`), or its example values are, because they are read once for a structure of the graph. | Call `agent.clear_schema(graph)` (or `agent.clear_schema()` for every graph). The next run reads it again. A query that uses a new name is still validated against the graph. |

## The cost of trying again

A question costs one LLM call when the first query passes, and at most `total_attempts + 1` calls (plus one when an empty answer on the filtered schema is retried on the full one). A query repaired without the LLM costs none. See [Cost](how-it-works.md#cost).
