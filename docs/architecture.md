# Architecture

text2cypher turns a question into a validated, read-only openCypher query for **FalkorDB**. It is an agentic
loop: generate a query with an LLM, validate it against the real graph, correct it with what the validator
found, escalate from a filtered to the full schema, and optionally execute the result.
Validation is done by [`cypher-guardrail`](https://pypi.org/project/cypher-guardrail/).

This document describes the current design and why it is the way it is. User-facing documentation is in
`README.md` and the Sphinx site (`docs/*.md` pages listed in `docs/index.md`).

## The loop

```
question ──► filter schema ──► generate ──► validate ─┬─ valid ───────────► (execute) ──► Result
                                   ▲                  ├─ invalid, budget left ─► correct ─┐
                                   │                  │                                   │
                          full schema ◄── escalate ◄──┤                      validate ◄───┘
                                                      └─ budget exhausted ──► failed Result
```

The loop is driven by explicit counters, not by inspecting a history list:

1. Generate with the **filtered** schema (`schema_scope = "filtered"`).
2. Validate. If the query has no issues, finish (executing it if `Config.execute`).
3. Otherwise correct and validate again, until `attempts_w_filtered` corrections have been spent; then the
   corrections use the **full** schema.
4. Continue until `total_attempts` corrections, then return a `failed` Result that still carries the last query
   and its issues.

The first query is not a correction: the LLM is asked at most `total_attempts + 1` times (plus one retry on the
full schema if it declines on the filtered one). Defaults are 3 and 5.

## Principles

- **Async only.** Every I/O step is a coroutine (schema read, validation, LLM calls, execution). A sync facade
  cannot simply wrap the core: a sync caller holds a sync client, and the `falkordb.asyncio` client is bound to
  the event loop that first used it. Supporting both would mean duplicating the pipeline or owning a loop and a
  client internally. The library takes a client from the caller and never creates one; use one `Text2Cypher` per
  event loop.
- **A plain loop, no graph framework.** The workflow has one cycle and about six states; an `async` loop with
  explicit state is easier to test and keeps LangGraph and LangChain out of the dependencies. The cost is no
  built-in diagram or checkpointing.
- **Read-only by construction.** The validator rejects any query that can write; execution uses `ro_query`, which
  FalkorDB itself rejects for a write. Both are always on.
- **Honest outcomes.** A query that fails validation is never returned as if it were fine: the status is
  `valid`, `unanswerable` or `failed`, and every attempt is kept.
- **Structured data, not parsed text.** The validator's issues carry typed fields; we read the fields and never
  match on description text, and a contract test pins the field names we depend on so an upstream change fails a
  test instead of degrading silently.
- **Nothing is sent to the LLM but names and value types.** The prompt has the question, the schema (labels,
  relationship types, properties with their value types, paths) and earlier attempts with their problems; no data
  from the graph and no example values.

## Package layout

Public API: what `text2cypher`, `text2cypher.llm` and `text2cypher.schema` list in `__all__`, pinned by
`tests/test_public_api.py`, which also fails if a module outside them lacks a leading underscore. Everything
else is internal. Dependencies point down, and `import-linter` enforces it in CI: `_pipeline` -> `_workflow` ->
(`_generation`, `_execution`, `_schema_source`) -> `_results` -> `_validation` -> (`llm`, `schema`, `_config`, `_errors`),
and the package never imports `evals`. Each type lives with its domain (there is no shared `types` module).

```
src/text2cypher/
├── __init__.py          the public API (re-exports, __all__, __version__); no logic
├── py.typed
├── _pipeline.py         Text2Cypher: gets the snapshot, runs the workflow, executes
├── _schema_source.py    SchemaSource: a new snapshot per read, or a kept one per graph (and the missing-graph error)
├── _workflow.py         run_workflow: the loop and the escalation policy
├── _config.py           Config
├── _results.py          Result, Attempt, Records, Status, SchemaScope
├── _execution.py        execute_cypher: read-only, with a timeout
├── _errors.py           GraphNotFoundError
├── llm/                 public: AsyncLLMClient, Completion, Usage, OpenAIClient, FakeLLM, Call
│   ├── _base.py  _openai.py  _fake.py
├── schema/              public: GraphSchema, Path, SchemaFilter, filter_schema, render_schema, schema_from_snapshot
│   ├── _model.py  _from_snapshot.py  _filter.py  _stopwords.py  _render.py
├── _validation/
│   ├── validator.py     QueryValidator: the three cypher-guardrail async validators
│   └── issues.py        Issue, Source, ValidationResult, drop_repeated_property_issues
└── _generation/
    ├── prompts.py       generator system prompt (rules built from lists) and the question message
    ├── conversation.py  Round, the turns of a conversation (chat_messages) and the same as one message (folded_message)
    ├── problems.py      the issues of a query worded for the LLM, one handler per issue code
    ├── extract.py       extract_cypher: first fenced block, else the trimmed reply
    └── generate.py      generate_cypher (a first query or a correction), Generation
tests/                   mirrors src/: llm/ schema/ validation/ generation/ trial/, and test_workflow,
                         test_config, test_pipeline, test_execution, test_examples, test_public_api;
                         support.py, conftest.py
examples/                quickstart.py (embedded FalkorDB, scripted LLM, one correction); run by a test
evals/                   the trial harness (not part of the package): dataset (types), datasets/ (movies, company),
                         replay, judge, metrics, trial, recorded/
docs/                    the Sphinx site (conf.py and the guides) and this document
```

## Components

### Pipeline (`_pipeline.py`)

`Text2Cypher(client, llm, config=None, schema_filter=filter_schema).run(question, graph)`:

1. Asks its `SchemaSource` for a `Snapshot` (`types=True`) of the graph: a new capture on every call, or, with
   `Config.schema_refresh_seconds`, a kept one (see Schema).
2. The source raises `GraphNotFoundError` (a public `ValueError`) before any LLM call if the snapshot has no labels and
   no relationship types: FalkorDB has no graph without data, so an empty snapshot means a missing graph or a typo,
   and nothing is kept for it.
3. Builds the `GraphSchema` from the snapshot and runs the workflow with a `QueryValidator` that answers from the same
   snapshot (hybrid mode), so the prompt and the validator cannot disagree.
4. If `config.execute` and the result is `valid`, executes the query and puts the rows or the error on the result.

Errors from the LLM or from the database while reading the schema or validating propagate; nothing is caught.

### Workflow (`_workflow.py`)

`run_workflow(question, schema, *, llm, validate, config, schema_filter=None)` is the loop. The validator is
passed as a callable, so the loop is tested without a database. State is explicit: the failed query to correct,
the number of corrections, and whether the filtered schema has been given up.

- A filtered schema exists only if a `schema_filter` is given and keeps less than the whole schema. A filter that
  keeps everything counts as no filter, and every attempt is `full`.
- A decline (an empty fenced block) on the filtered schema may be the filter's doing, so the same request (the
  first query, or the same correction) is made again with the full schema, once, without using a correction;
  later corrections then use the full schema. Only a decline on the full schema is `unanswerable`. The empty
  attempt stays in `Result.attempts` with its usage.
- An empty query is never validated: the validator would call it a syntax error and the loop would spend its whole
  budget on it.
- `Config` rejects negative values and `total_attempts < attempts_w_filtered`.

### Results (`_results.py`)

`Result` has `status`, the last `query`, every `Attempt` (query, `ValidationResult` or `None` for an empty query,
schema scope, `Usage`), `usage`, `records` and `execution_error`. `Result.usage` is the sum of the attempts' usage,
or `None` if any attempt did not report it. `Issue`, `ValidationResult` and `Attempt` are hashable (the dict inside
an `Issue` is left out of its hash); a `Result` is hashable unless it holds `records`.

### Validation (`_validation/`)

- Uses `AsyncSyntaxValidator`, `AsyncSchemaValidator` and `AsyncPropertiesValidator` from `cypher-guardrail`.
  Their return shapes differ and are normalised: syntax gives `(bool, metadata)`, schema `(score, metadata)`,
  properties `(score | None, metadata)`.
- Every metadata item is a typed dict from `cypher_guardrail.issues`: a `code`, a `description` and fields per
  code (`kind`/`name`/`properties` for a property access, `suggested_path` for a path, `parameters`, `variables`,
  `index`, ...). Our `Issue` wraps it, adding its `source` validator (`syntax`, `schema`, `properties`), so it
  stays JSON-able and type-narrowable. We compare codes against the library's constants, never the strings.
- **All three validators always run**, even when the syntax is invalid: they can still report real problems
  (an unknown label) elsewhere in the query, so one pass reports every kind of problem.
- **Validity** is `syntax_ok and not issues`, never derived from scores. A schema score of 1 or a properties score
  of `None` with no issues means "nothing to check", not "correct". Scores stay on `ValidationResult` as metrics.
- **Two optional checks.** `Config.check_types` (property type mismatches, aggregates of non-numbers) and
  `Config.check_performance` (cartesian product, unlabeled scan, unbounded path) are passed to the validators. `check_types`
  is on: it reported nothing on 33 recorded real-model runs and the 40 gold queries of both trial graphs, and what it reports
  fails or returns nothing. `check_performance` is off: each of its warnings can be what the question needs (a shortest path
  between two nodes is a cartesian product by design, and our own prompt example is one). A warning from either is corrected
  like any other issue.
- **No severity levels.** Every reported issue triggers a correction. The candidates for a "warning" are all
  defects here (`ParameterNotProvided`: we execute without parameters; `UnknownIndex`: the query would return
  nothing or fail). Add severity only if a real non-blocking case appears.
- A write query is reported as `cypher_guardrail.Error.WriteQuery` (source `syntax`) and is not valid.
  `QueryValidator` does not expose `read_only`: rejecting writes is the right default for generated queries.
- **One shared `AsyncFalkorDBBackend`**, and the three validators run concurrently in an `asyncio.TaskGroup`
  inside `with backend.snapshot():`, so they share the schema cache and the concurrency limit. A failure cancels
  the others before the snapshot is released, and the first error is re-raised unwrapped.
- `QueryValidator(client_or_graph, max_concurrency=8).validate(query, graph=None)` returns
  `ValidationResult(issues, syntax_ok, schema_score, properties_score)`; it raises `ValueError` when there is no
  graph to validate against.
- **One root cause, one issue.** An unknown label or relationship type also produces a property issue for the
  same name. `drop_repeated_property_issues` drops a property issue whose `(kind, name)` matches an unknown label
  (`node`) or unknown relationship type (`relationship`), so a real label that shares its name with an unknown
  relationship type keeps its issue.

### Schema (`schema/`)

- `GraphSchema` is a data structure separate from reading and rendering, so filtering works on structure, not
  strings: `node_properties` (label -> property -> value types), `relationship_properties` (the same per type) and
  `paths` (a tuple of one-hop `Path(source, type, target)`) and `indexes` (a tuple of `Index(entity, property, type)`,
  with `type` `RANGE`, `FULLTEXT` or `VECTOR`, read from the snapshot; the filter keeps those of what it keeps).
- `schema_from_snapshot(snapshot)` builds it from a `cypher-guardrail` `Snapshot` captured with `types=True`.
  FalkorDB has no declared schema, so the snapshot defines it by the data: names, properties with value types,
  and edges as label-set triples. Each edge expands into one `Path` per label pair (a multi-label node contributes
  a path per label). Value types need a full scan (`types=True` cannot be combined with `sample=`), so nothing is
  sampled. A snapshot captured without types is rejected (`ValueError`); a graph with no data gives an empty schema.
  The snapshot also feeds the validators (cypher-guardrail's hybrid mode): they answer lookups from it and ask the
  database only to plan the query (`EXPLAIN`) and to confirm a problem before reporting it. Checked on the test graph:
  identical issues and scores for every kind of issue, and a query that is fine costs one request instead of up to ten.
- **Keeping the schema.** `Config.schema_refresh_seconds=None` (the default) captures on every run. A number keeps one
  `AsyncLiveSnapshot` per graph: when the time is up, `Text2Cypher` awaits its `refresh()`, which compares a fingerprint
  (names, indexes, node and relationship counts) and scans again only if it differs. The live snapshot's own background
  refresh is switched off (it is never due), so no task outlives the event loop. A graph that did not exist is not kept,
  so one created after the error is found. The fingerprint is a heuristic, and between checks a change is missing from the
  prompt; it is not missing from validation, because a problem is confirmed against the database before it is reported.
- Names containing a backtick are skipped: FalkorDB has no escape for one, so no query can use such a name, and
  skipping them also removes any chance of injecting through a name read from the database. Other names are quoted
  only through `quote_identifier`, and only when rendered.
- `render_schema` produces the Neo4j-style text the generator prompt carries: `Node properties:`,
  `Relationship properties:`, `The relationships:`, and `Indexes:` when there are any. Types are upper-cased and joined with `|`; names that are not
  plain identifiers are backtick-quoted; sections with nothing to say are left out.
- **Filtering** (`filter_schema(schema, question)`, the default `schema_filter`) matches the question's words to
  the words of label and relationship-type names: Snowball stems (`snowballstemmer`), camelCase and underscore
  splitting, case-insensitive. Stopwords are left out of names (so "in" cannot find `IN_GENRE`) unless a name is
  only stopwords. A matched label keeps the paths touching it, a matched type keeps its paths, and every label and
  type on a kept path stays with its properties, so the LLM can use the far end. Property names are not matched. A
  question that matches nothing keeps the whole schema, so a miss costs nothing; the lexical filter does not know
  that "people" means `Person`. A `SchemaFilter` is a plain synchronous callable `(schema, question) -> schema`.

### LLM (`llm/`)

- Two stateless protocols. `AsyncChatClient` has `chat(system, messages) -> Completion(text, usage)`, with
  `messages: Sequence[Message]` (`Message(role, content)`, roles `user` and `assistant`, first and last from the user) and
  `usage: Usage | None`; the library calls it when a client has a callable `chat`. `AsyncLLMClient` keeps its one method,
  `complete(system, user)`, so a client written for 1.x still works: it is given the same conversation as one user
  message. Both are public and additive, so nothing breaks. A client with both gets `chat`; an attribute named `chat`
  that is not callable (an SDK client's namespace) does not count. The choice is made in `generate_cypher`, not in an
  adapter, because the one-message form lists the attempts and ends with the instruction once, which a generic
  adapter over turns could not do without parsing the prompt text.
- `OpenAIClient(model, *, client=None, temperature=0)` uses `AsyncOpenAI` (Chat Completions); pass your own client
  for an OpenAI-compatible server, and `temperature=None` for a model that only accepts its own. `FakeLLM` replays
  scripted replies and records its calls, for tests.

### Generation (`_generation/`)

- **Extraction**: the first fenced block anywhere in the reply, else the whole reply trimmed (models add prose around
  fences, and validation catches what is left). A language tag (any word on the fence's own line, or `cypher`,
  `opencypher`, `cql` on a one-line block) is dropped; Windows line endings are accepted. An empty block means the
  LLM declined. A reply with no fence at all is taken as the query: a prose refusal then fails validation and the
  loop recovers, and a heuristic that declined it could decline a valid query.
- **Prompts** are our own wording. The generator system prompt has the task, a short FalkorDB rules block, two
  examples on another schema and the rendered schema; the user message is only the question, so the question is
  never in the system prompt and the system prompt has a stable prefix. A correction reuses that system prompt and
  continues the conversation (`_generation/conversation.py`): the question, then for every failed query the LLM wrote
  that query as an assistant turn in a fence and a user turn that lists its issues and asks for a corrected query
  (the failed query is not quoted again, it is the turn before). Each round keeps its issues, not their wording,
  and the wording is made for each request with that request's schema, so the "did you mean" hints always name what the
  system message lists. When the schema changes from filtered to full, the system message is swapped and the turns
  stay; the first user turn after the change opens with a line saying that the schema now lists the whole graph (once,
  and only when a query was written with the filtered one: a decline on the first query has nothing to explain). A
  declined reply (an empty block) is never a turn: it is asked again with the same turns and the full schema.
  A client with only `complete` gets the conversation as one user message: the question, the note if any, "Earlier
  attempts, oldest first" with each query and its problems, and the instruction once at the end.
- **Check section**, after the rules and before the reply format: the LLM is asked to check the query before it answers (valid syntax; every label, relationship type and property in the schema as spelled; every
  relationship in the direction the schema shows; read-only; only what the question asks for) and not to write the checks in its reply. The validators run the same checks afterwards, but a mistake the LLM catches costs
  no correction. About 55 words in every system prompt; not measured (both trial graphs were at the ceiling), so the changelog says so. With `Config.use_parameters` it also checks
  that every `$parameter` has a value in the `CYPHER` header, separated by spaces and not commas (the commonest problem in that mode), and the format example is shown with a header, so
  the example never contradicts the rule that values go in parameters (default mode is byte for byte what it was; both examples are run on FalkorDB by `test_prompt_rules.py`).
- **Rules block**: read-only; write values inline and never `$param` (a bare `$n` is `ParameterNotProvided`), unless
  `Config.use_parameters` swaps that rule for one that asks for the values in a `CYPHER` header (see Parameters); for a
  shortest path between two nodes use `algo.SPpaths`, since FalkorDB supports `shortestPath` only in `WITH` or
  `RETURN`; return readable fields, not a bare path or whole nodes. Every rule is checked on an embedded FalkorDB by
  `tests/generation/test_prompt_rules.py`, and so is every query the prompt shows. Not-equal is written `<>`
  (`!=` is a syntax error in FalkorDB).
- **Parameters** (`Config.use_parameters`, off): the values go in a `CYPHER` header, `CYPHER name='Ann' year=1990 MATCH (p:Person {name: $name}) ...`. Checked on FalkorDB:
  pairs are separated by spaces (commas are a syntax error), strings, numbers, lists and `LIMIT $n` work, text with an apostrophe needs double quotes, a `$x` not in the header is
  `ParameterNotProvided`, and the whole query validates and runs read-only as one string, so `Result.query` needs no splitting. The corrector's wording for a missing value says
  to give it one in the header. FalkorDB can plan such a query once and reuse the plan; that is the whole benefit, so the option is off.
- **User-defined functions**: the snapshot's `udfs` become `GraphSchema.functions`, listed in the prompt with the rule to call only those listed; an unknown dotted function is reported
  by both validators and worded with the closest listed name and the list.
- **Index rules**, added only for the kinds of index the schema has (which is also listed under `Indexes:`). Checked on
  FalkorDB with `EXPLAIN`: `=` (also the inline `{name: 'x'}` form), `<`, `<=`, `>`, `>=`, `IN` and an `OR` of
  indexed properties use the index; `<>`, `NOT`, `STARTS WITH`, `CONTAINS`, a function (`toLower`) and arithmetic on the
  property do not. So a RANGE index gets "compare directly; the others read every node with the label, use them only
  when the question needs them". A FULLTEXT index gets the `db.idx.fulltext.queryNodes` example, for word searches only.
  A VECTOR index gets "do not call `db.idx.vector.*`": a vector query is valid (the validator does not stop it) and the
  model has no embedding to give it.
- The question is data, never instructions: anything that is not a request for a query gets an empty block.
- A correction turn words each issue from its structured fields and the schema of the request
  (`_generation/problems.py`), not from the validator's description, which is written for developers (some is
  FalkorDB's own error text): an unknown label, relationship type or property says so plainly and offers the closest
  name as a "did you mean" when it is a likely typo (very similar, or fairly similar and starting alike: `rate` for
  `rating`, not `Client` for `City`); a property that a label lacks says where it does exist ("`hours` is a property of
  the relationship type `ASSIGNED_TO`") and lists the label's own properties; a path that does not exist lists where its
  relationship type does go; a `$parameter` is told to be written in the query; a write is "The query can modify the
  graph. Only read queries are allowed."; a wrong-direction or variable-length issue keeps its description and adds its
  `suggested_path` unless the description already has it. An unknown-property issue is dropped when a label issue names
  the same property. Anything else keeps the description.

### Execution (`_execution.py`)

- Off by default (`Config.execute = False`).
- `execute_cypher(graph, query, *, timeout_ms=None) -> Records` runs `ro_query` with a timeout: 10 s by default
  (`Config.execute_timeout_ms`), `None` for no limit, because a generated query can be expensive and an unlimited
  default would let one hang the caller.
- Only a `valid` result is executed, after the workflow. `Result.records` is `Records(columns, rows)`, with the
  client's own values (`Node`, `Edge`, `Path`). Any `redis.exceptions.RedisError` (a rejected or failed query, a
  timeout, a lost connection) goes to `Result.execution_error` and the status stays `valid`, so the LLM calls are
  never lost; anything else is a bug and propagates.
- A runtime failure is not fed back to the corrector: it would change how corrections are counted, and the
  validator already catches most such errors. There is no cap on returned rows: the timeout bounds time, not
  memory, and a result-size limit belongs on the server.

## Tooling and constraints

- Python `>=3.12` (CI matrix 3.12, 3.13, 3.14): the embedded FalkorDB used by the tests needs 3.12, and nothing needs
  a newer feature. `py.typed` is shipped; the classifiers say production/stable, Python 3.12-3.14, typed.
- uv for everything (`uv add`, `uv run`); ruff for lint and format (rules B, I, UP, SIM, C4, C90 with a
  complexity limit of 10, PTH, RET, RUF, ASYNC, PT, TID, TC, FURB); ty for types; import-linter for the layers above; pytest (importlib import mode, strict markers); coverage with branches. See
  `CLAUDE.md`.
- Dependencies: `cypher-guardrail>=1.4.0`, `falkordb`, `openai`, `redis` (for `RedisError`), `snowballstemmer`.
  Dev: `falkordblite`, `anyio`, `pytest`, `ruff`, `ty`, `coverage`, `import-linter`, `httpx2` (the OpenAI client tests run the real
  SDK over a mock transport). Docs group: `sphinx`, `myst-parser`, `furo`, `sphinx-copybutton`.
- Documentation: Sphinx with MyST Markdown, Furo and autodoc (Google docstrings through napoleon), built by Read
  the Docs (`.readthedocs.yaml`, `fail_on_warning`) and by a CI job with `-W`. The build is nitpicky: every type in
  a signature must resolve (third-party types are listed in `nitpick_ignore_regex`). `docs/conf.py` excludes this
  document and the other internal notes; `examples/quickstart.py` is included in the usage page and run by
  `tests/test_examples.py`.

## Test infrastructure

- `tests/conftest.py` provides a session-scoped async client (`falkordb_client`, a regular
  `falkordb.asyncio.FalkorDB`) attached to the unix socket of an embedded FalkorDBLite server. The sync embedded
  client (`redislite.falkordb_client.FalkorDB`) owns the server so it is stopped cleanly with a save; FalkorDBLite's
  own `AsyncFalkorDB` is not used because of an unawaited-`shutdown` bug (FalkorDBLite issue 61).
- Per-test graphs are deleted afterwards (`movies_graph`, `film_graph`, an empty `new_graph`); the server is shared
  because starting and stopping it is slow. Every test that uses `falkordb_client` is marked `db` automatically,
  so `pytest -m "not db"` is the fast run. `tests/support.py` has the builders shared by the tests (a fenced
  `reply`, `make_issue`, `invalid`/`VALID`, `ScriptedValidator`, a small hand-built schema).
- Async tests are marked `@pytest.mark.anyio` (module-level `pytestmark`).
- `evals/` (not part of the package) is a trial harness: it replays questions through the real `Text2Cypher.run`
  with replies written by a model or a person, on a throwaway embedded server with a film graph, and judges the
  final query by the rows it returns against a gold query (extra columns are accepted, a duplicate row counts).
  `tests/trial/` checks it with the gold queries, and `tests/trial/test_recorded_trial.py` replays the recorded
  replies of a real trial (`evals/recorded/`) and pins the outcome of each question, which guards the filter, the
  validator, the loop and the execution against regressions (it does not pin prompt wording; `tests/generation/`
  does). See `evals/README.md`.
- A third trial compared the corrector's old and new wording on the company graph (a schema-shaped graph, 16 seeded
  flaws, Haiku as the model, judged by validation): both wordings healed every question, 1.1 corrections each, so
  that model and set do not separate them. The decline on the filtered schema and the retry on the full one were what
  mattered for the one question whose path needed a name the filter had dropped.
- What the first two trials showed (two rounds, a small model, a small graph): the loop and prompts work; every decline in
  the trials was an empty block; the filter kept every name the gold queries needed except where a question's words
  never match a name ("people" for `Person`), which fell back to the full schema. Not measured: a weaker model, a
  large schema, how much the filter prunes, how often the decline retry fires.

## Not built yet

Each item says what would bring it back.

**Planned**
- **LLM-based schema filtering**: the LLM picks the relevant labels, relationship types and properties, behind the
  same `SchemaFilter` interface so strategies can be compared. It needs an async filter type. Trigger: a measured
  filtered-schema miss rate on a large graph that the lexical filter's full-schema fallback does not absorb.
- **Auto-fix of simple typos without the LLM** (it changes how attempts are counted). The corrector wording itself is
  built; see Generation.
- **Whether structured issues can inform escalation**: an unknown label that exists nowhere in the graph will not
  be fixed by the full schema. Untested.
- A sync facade, if demand appears (see Async only).
- `run_many(questions, concurrency=N)` for dataset evaluation.

**Left out of the prompt, each until evaluation or a feature needs it**
- `toLower()` guidance beyond the index rule: a function on an indexed property prevents index use, but a
  case-insensitive match is often what the question means. Decide with the harness.
- More path guidance (`algo.SSpaths`, weighted paths, `pathCount`, bounded variable-length paths); `LIMIT`
  guidance (adding a limit the question did not ask for changes the answer).
- Few-shot examples generated from the schema; the schema as inline Cypher patterns instead of the Neo4j-style
  text; the schema in the user message instead of the system prompt (ours keeps a stable prefix for provider
  prompt caching); a numbered `code: description` error format for the corrector.
- Example values in the schema (they need separate queries, and send graph data to the LLM provider).
- Dynamic skills (the model reads guidance through tool calls: needs a tool-calling protocol, which a stateless
  `chat(system, messages)` does not have); answer synthesis.
- A model without a system role: an option on `OpenAIClient` that sends one user message.

**Other**
- Embedding similarity for the lexical filter (a heavy dependency, would be an extra) and property-name matching:
  when the filter's miss rate justifies them.
- Feeding a runtime failure or an empty result back to the corrector; a row limit; converting rows to plain Python.
- Metrics for the harness: success rate, self-healing rate (failures fixed by a correction), validation failures
  by type.
