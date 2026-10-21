# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- `cypher-guardrail` is now `>=1.5.1` (was `>=1.4.0`).

## 2.0.0 - 2026-10-21

The LLM client protocol takes tools, which is the one breaking change (see `MIGRATION.md`). It brings skills the LLM can read, an LLM-based schema filter, the OpenAI SDK client passed as it is, and the reason a reply ended; the filter and skills were measured with Haiku and Sonnet and both stay off by default.

### Added

- A measurement of the LLM schema filter and of skills, with Haiku and Sonnet (one run per cell, in the `evals/` harness, replies in `evals/recorded/measurement/`): on a 36-label schema the LLM filter chose the right part on the first try for 20 and 19 of 20 questions against 12 and 11 for the word-matching filter, and skills lifted Haiku from 10 to 12 of 15 questions and made no difference for Sonnet. Both stay off by default; the README has the recommendation.
- `Completion.finish_reason` (`"stop"`, `"length"`, `"content_filter"`, `"refusal"`, or the provider's word; `None` if not known), which `OpenAIClient` fills from the API (a message with a `refusal` and no text is a `"refusal"`), and `Attempt.finish_reason`, on every attempt that asked the LLM.
- `Config(use_skills=True)` (off by default): the LLM is offered skills, short guides to parts of FalkorDB's Cypher that a model gets wrong without them: `graph-algorithms` (PageRank, betweenness, components, communities, BFS), `path-finding`
  (`algo.SPpaths` and `algo.SSpaths` with their options, variable-length patterns, where `shortestPath` works) and `functions` (the functions FalkorDB has, and what it lacks or spells differently, such as `!=`, `=~`, `datetime()` and `COUNT {}`).
  The prompt lists them with a line each, and a tool, `read_skill`, reads one when a question needs it (up to three rounds, then the model must answer). **Each read is another call to the model.** What was read is in the prompt of the next request, so a
  correction does not read it again. A client that cannot be offered tools has every skill in the prompt instead. `Attempt.skills` lists the skills read for a query and their tokens are in `Attempt.usage`. The examples of every skill, and each claim that something is missing, are run against a
  real FalkorDB server whenever the library is tested. The path-finding skill is adapted from a FalkorDB skill, under its MIT license, whose notice ships with the package (`THIRD_PARTY_NOTICES`).
- `Tool` and `ToolCall`, in the shape that the OpenAI SDK, MCP and the other tool-calling APIs share (a name, a description and a JSON Schema; an id, a name and arguments as a dict), `Completion.tool_calls`, and `Message`
  turns with `tool_calls` and a `"tool"` role carrying `tool_call_id`. `OpenAIClient` sends the tools as functions, answers every call id, and parses the model's calls (`OpenAIClient(tools=False)` for a server that cannot
  call tools); `FakeLLM` records the tools and the choice on each call and can script a tool call.
- `MIGRATION.md` (also in the docs): what to change when moving from 1.x.
- An `openai.AsyncOpenAI` (also `AsyncAzureOpenAI`, and a client for a compatible server) can be passed as the `llm` of `Text2Cypher` with `model=`, without wrapping it in `OpenAIClient`: `Text2Cypher(client, AsyncOpenAI(), model="...")`.
  An SDK client has no model of its own, so `model=` is required for it and is an error with any other client; the synchronous `openai.OpenAI` is refused. `LLMSchemaFilter` takes `model` too: with its own
  SDK client it names the model, and with `llm=None` it asks the main client with another model (a cheaper one), through the new `OpenAIClient.with_model`.
- `LLMSchemaFilter(llm=None, min_entities=20)`: a schema filter that asks a model which labels and relationship types a question needs, instead of matching its words to names. Choosing it replaces the
  lexical filter, and it is shown the whole schema (names, property names and paths, no types or examples). It uses the main model unless given its own, perhaps a cheaper one. A reply that cannot be used, or a call that
  fails, keeps the whole schema, and a schema below `min_entities` is kept whole without asking. `Result.filter_usage` has what it cost and `Result.usage` includes it. Filters may now be `async`: `FilterResult` and `AsyncSchemaFilter`
  are new, and `restrict_schema` is the public helper that keeps the named labels and types with their paths and what goes with them.

### Changed

- **Breaking, for a client of your own:** `AsyncChatClient.chat` takes the tools a model may be offered: `chat(system, messages, tools=None, *, tool_choice="auto")`. A client with the old two-argument `chat` keeps
  working at run time, because tools are offered only to a client that takes them (it says so with `supports_tools`, or its `chat` takes `tools` or keyword arguments), but a type checker reports the new signature.
  `Completion`, `Message` and `Call` have more fields (with defaults), so code that unpacks them into a fixed number of names must read the fields by name. See `MIGRATION.md`.
- A reply from the LLM with no text at all (or only white space) is no longer taken as a decline. It is a call that returned nothing, so the same request is made once more, exactly as it was, and it costs no correction; a second empty reply in
  a row ends the run as `failed`, with no query, and not as `unanswerable`. `Attempt.empty_reply` marks each such attempt. An explicit empty fenced block is still a decline, as before. Before, a provider that returned nothing gave `unanswerable`.
  Why the reply ended decides it when the client says: a `refusal` is a decline, `length` and `content_filter` are not asked again (the same request would hit the same limit or filter), and anything else, or not known, is asked again once.

## 1.2.0 - 2026-10-21

A correction is now a conversation that carries every earlier attempt, a query that FalkorDB fails when it runs is corrected, a misspelled name or a broken parameter header can be fixed without the LLM, and the LLM is shown example values of your data. Several defaults and behaviours change, so read **Changed** before you upgrade: the limits are higher, example values are sent to the LLM provider unless you pass `example_values=0`, `execute=True` can now end as `failed`, and a failure of the LLM client reaches you as an `LLMError`.

### Added

- `Text2Cypher.clear_schema(graph=None)` forgets what is kept of the schema of one graph or of all of them (the snapshot, and the example values, which are otherwise read once for a structure of the graph), so the next
  run reads it again. A new guide, "When a question fails", says what to look at and what to try for each outcome.
- With `use_parameters`, a `CYPHER` header with commas between the pairs, or without the `CYPHER`, is repaired without the LLM: FalkorDB rejects both as a syntax error, which used to cost a correction. The query is
  validated again and, if it passes, is the answer at no cost in LLM calls (`Attempt.auto_fixed`). A comma inside a string or a list is kept. It changes nothing about what a query means, so it has no option.
- `OpenAIClient(timeout=120.0, max_retries=None)`: how long a call waits for a reply, and how many times the SDK retries a failing one. The SDK's own timeout is 10 minutes, which a loop of up to a dozen calls
  could multiply into hours on a hung connection. The timeout applies to a client you pass too, and `None` leaves that client's own.
- `Text2CypherError`, the base of the errors the library raises, and two new ones for a run that stops in the middle: `LLMError` (the LLM client raised) and `DatabaseError` (FalkorDB failed while a query was being
  validated). Both carry `question`, `attempts` and `usage` of the attempts made before the failure, which were paid for and used to be lost, and the original exception as `__cause__`. `FakeLLM` raises a reply that is an exception.
- `Config(autofix_typos=True)` (off by default): a label, relationship type or property that the validators report as unknown is replaced, without the LLM, by the one name of the graph it is a likely typo of
  (never inside a string or a comment), and the query is validated again, up to three times. If it passes it is the answer, at no cost in LLM calls or corrections; if not, it is dropped and the LLM corrects
  the query it wrote. `Attempt.auto_fixed` marks the fixed attempt, which has `Usage(0, 0)`.
- `Config(example_values=3)`: the schema the LLM is shown has up to three example values of each property of a plain type (`title: STRING (e.g. 'Alien', 'Heat')`), so it writes a value the way the graph does,
  and the prompt gets a rule to do so. **They are rows of your graph and go to the LLM provider: pass `example_values=0` to send none.** They are read with one small read-only query per property (with a timeout),
  once for each structure of the graph. `GraphSchema.node_examples` and `relationship_examples` hold them.
- `AsyncChatClient` and `Message`: a client with a `chat(system, messages)` method is given a correction as the turns of a conversation. `OpenAIClient` and `FakeLLM` have it, and a
  `FakeLLM` call records every turn (`Call.messages`). A client with only `complete(system, user)` keeps working: it is given the same conversation as one user message.

### Changed

- An exception raised by your LLM client during `run` now reaches you as an `LLMError`, with the original as `__cause__`, and a failure of FalkorDB while validating as a `DatabaseError`. Code that catches a provider's own
  exception (for example `openai.RateLimitError`) around `run` has to catch `LLMError` and look at `__cause__`. `GraphNotFoundError` is unchanged, and is still a `ValueError`.
- The default limits are higher: `attempts_w_filtered=5` (was 3) and `total_attempts=10` (was 5). A question that cannot be fixed now costs up to 11 LLM calls (was 6) before it ends as `failed`.
  Pass `Config(attempts_w_filtered=3, total_attempts=5)` for the old limits.
- With `Config(execute=True)`, a query that FalkorDB rejects or fails when it runs (a type error on a value, a division by zero) is no longer returned as `valid` with `execution_error`: the error is sent
  back to the LLM as the next turn, and the corrected query is validated and run, which costs a correction like any other. When the corrections run out on such queries the status is now `failed`.
  A timeout or a lost connection still ends the run as `valid` with `execution_error`, because it says nothing about the query. `Attempt.execution_error` has the error of each attempt. FalkorDB's error
  message is sent to the LLM and can name a value from your data.
- The system prompt asks the LLM to check its query before it answers: valid syntax, every name in the schema, the direction of every relationship, read-only, and only what the question asks for. Whether it
  saves corrections has not been measured: both test graphs were already at the ceiling. With `use_parameters` it also checks that every `$parameter` has a value in the `CYPHER`
  header, and the format example in the prompt is shown with a header, so it no longer contradicts the rule.
- A correction now carries every query the LLM wrote so far with what was wrong with it, not only the last one, so it does not repeat a mistake it was told about. Each failed query is an
  assistant turn followed by a user turn with its problems, and the failed query is no longer quoted again in the turn about it.
- When the schema changes from the filtered to the full one, the earlier turns are kept and the first turn after the change tells the LLM that the schema now lists the whole graph.
- `Call` (what `FakeLLM` records) has a third field, `messages`, and `user` is the text of the last user turn.

## 1.1.0 - 2026-10-20

### Added

- The user-defined functions of the FalkorDB instance (`GraphSchema.functions`) are in the schema the LLM is shown, with the rule to call only those listed; an unknown
  function is reported with the closest listed name and the list.
- `Config(use_parameters=True)`: the LLM puts the values of the question in a `CYPHER` header and uses parameters, which FalkorDB can plan once and reuse. `Result.query` is
  still one string that runs as it is. Off by default.

### Changed

- `Config.check_types` is now on by default: a property compared with a value of another type, and `sum`, `avg` and the like of a property that holds no numbers, are reported and
  corrected, because such a query fails or returns nothing. It reported nothing on recorded real-model runs and on the gold queries of both trial graphs. Pass
  `Config(check_types=False)` for the old behaviour.

## 1.0.0 - 2026-10-20

The first stable release. From here the public names (`text2cypher.__all__`, `text2cypher.llm` and `text2cypher.schema`) follow [semantic versioning](https://semver.org/): a breaking change to them needs a new major version. The prompts and the way the schema filter matches are not part of that promise and can change in a minor version. They have been tried with Claude Haiku on two small test graphs; see "Known limits" in the README.

### Added

- `Config(check_types=True)` reports a property compared with a value of another type and `sum`, `avg` and the like of a property that holds no
  numbers, and `Config(check_performance=True)` reports a cartesian product, a scan of every node and a path with no upper bound. Both are off by
  default, because each can report a query that is fine, and what they report is corrected like any other problem.
- The LLM is told about the new problems in words it can act on: an undefined variable, an unknown function (also a user-defined one that the
  instance does not have), a `LIMIT` or `SKIP` that is not a positive integer, and the type and performance problems above.

### Changed

- Requires cypher-guardrail 1.4.0 (it was 1.3.0). Validating a query that is a read no longer runs it for longer than a millisecond: a
  query of three cartesian products over 3000 nodes took more than a minute to validate before.

## 0.1.0 - 2026-10-20

The first release. It is alpha: the public names are pinned by a test, but the prompts, the defaults and the schema filter are likely to change as they meet more models and larger graphs.

### Added

- `Text2Cypher.run(question, graph)`: filter the graph's schema by the question, ask an LLM for a read-only
  openCypher query, validate it against the graph, correct it with the validator's findings, and escalate from
  the filtered to the full schema. The result says whether the query is `valid`, `unanswerable` or `failed`, and
  records every attempt with its schema and token usage.
- Optional, read-only execution of the final query with a timeout (`Config(execute=True)`).
- A small LLM protocol (`AsyncLLMClient`) with an OpenAI client (`system_role=False` for a model without a system role) and a scripted `FakeLLM` for tests.
- `GraphNotFoundError`, raised by `run()` before any LLM call when the graph does not exist.
- The schema carries the indexes of the graph (`text2cypher.schema.Index`), and the prompt lists them and adds a rule for each kind that
  exists: compare indexed properties directly, search words with the full-text procedure only for word searches, never query a vector index.
  The prompt also says that not-equal is `<>`.
- Problems are put to the LLM from the structured fields of each issue and the schema: "did you mean" hints for likely typos, where a property
  or a path does exist, and plain wording for unknown names and parameters.
- The validators answer from the snapshot the prompt was written from, so a query that passes costs one request to
  FalkorDB, and `Config(schema_refresh_seconds=...)` keeps a graph's schema between runs.
- Documentation built with Sphinx for Read the Docs (`docs/`, `.readthedocs.yaml`): a usage guide, how it works,
  the schema and its filter, LLM clients, the checks, and an API reference generated from the docstrings.
- A trial harness (`evals/`) that replays questions through the real pipeline with replies written by a model or a person.
