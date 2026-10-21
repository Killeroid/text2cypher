# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `Config(example_values=3)`: the schema the LLM is shown has up to three example values of each property of a plain type (`title: STRING (e.g. 'Alien', 'Heat')`), so it writes a value the way the graph does,
  and the prompt gets a rule to do so. **They are rows of your graph and go to the LLM provider: pass `example_values=0` to send none.** They are read with one small read-only query per property (with a timeout),
  once for each structure of the graph. `GraphSchema.node_examples` and `relationship_examples` hold them.
- `AsyncChatClient` and `Message`: a client with a `chat(system, messages)` method is given a correction as the turns of a conversation. `OpenAIClient` and `FakeLLM` have it, and a
  `FakeLLM` call records every turn (`Call.messages`). A client with only `complete(system, user)` keeps working: it is given the same conversation as one user message.

### Changed

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
