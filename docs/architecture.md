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

0. **Filter.** `schema_filter` keeps what the question needs of the schema: the lexical `filter_schema` by default, an `LLMSchemaFilter` that asks a model instead, or a function of your own, sync or async (see Schema). A
   filter that keeps the whole schema counts as no filter, and every attempt is `full`. What a filter cost is `Result.filter_usage`.
1. **Generate** with the **filtered** schema (`schema_scope = "filtered"`). With `Config.use_skills` the model may first read skills (up to three rounds of tool calls, see Reading skills); that is part of the same request.
   - An explicit empty fenced block is the LLM **declining**. On the filtered schema the same request is made again with the full one, once, at no cost in corrections; on the full schema the result is `unanswerable`.
   - A reply with **no text at all** is a call that returned nothing, not a decline: the same request is made again once, exactly as it was; two in a row end the run as `failed` with no query (ADR-047).
2. **Validate.** If the query has no issues, finish (executing it if `Config.execute`).
3. Otherwise **correct** and validate again, until `attempts_w_filtered` corrections have been spent; then the corrections use the **full** schema. A correction is the next turn of a conversation that carries every earlier
   attempt (see Generation).
4. Continue until `total_attempts` corrections, then return a `failed` Result that still carries the last query and its issues.

With `Config.autofix_typos`, and for the parameter header with `Config.use_parameters`, step 2 has a stage before a correction is asked for (see Typos and the header). With `Config.execute`, a query that has no issues is run, and a
failure of the database is a correction too (see Execution). If the LLM client or the database raises in the middle, the run ends with an `LLMError` or a `DatabaseError` that keeps the attempts and the tokens so far (see Errors).

The first query is not a correction: a question costs at most `total_attempts + 1` requests (plus one when a decline on the filtered schema is asked again on the full one, and one for each empty reply), and each request is
one call, up to four with skills. Defaults are 5 and 10 (ADR-046).

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
  `valid`, `unanswerable` or `failed` (which can also mean two empty replies and no query), and every attempt is kept.
- **Structured data, not parsed text.** The validator's issues carry typed fields; we read the fields and never
  match on description text, and a contract test pins the field names we depend on so an upstream change fails a
  test instead of degrading silently.
- **What goes to the LLM is stated, and the data in it can be turned off.** The prompt has the question, the schema (labels,
  relationship types, properties with their value types, paths), earlier attempts with their problems, and, by
  default, up to three example values of each plain property (`Config.example_values`, `0` sends none). Those values
  are rows of the graph that go to the LLM provider, so the docs and the option say so. FalkorDB's error text for a
  query that fails when it runs can name a value too. The rows a query returns never go to the LLM.

## Package layout

Public API: what `text2cypher`, `text2cypher.llm`, `text2cypher.schema` and `text2cypher.skills` list in `__all__`, pinned by
`tests/test_public_api.py`, which also fails if a module outside them lacks a leading underscore. Everything
else is internal. Dependencies point down, and `import-linter` enforces it in CI: (`_pipeline`, `skills`) -> `_workflow` ->
(`_generation`, `_graph`) -> `_results` -> `_validation` -> (`llm`, `schema`, `_config`, `_errors`, `_skills`),
and the package never imports `evals`. Each type lives with its domain (there is no shared `types` module).

```
src/text2cypher/
├── __init__.py          the public API (re-exports, __all__, __version__); no logic
├── py.typed
├── _pipeline.py         Text2Cypher: gets the schema, runs the workflow (which validates and executes), clear_schema
├── _workflow.py         run_workflow: the loop and the escalation policy
├── _config.py           Config
├── _graph/              talking to FalkorDB (ADR-053)
│   ├── schema_source.py SchemaSource: the schema and snapshot, new per read or kept per graph (and the missing-graph error)
│   ├── examples.py      sample_examples: a few example values of each plain property, read-only, with a timeout
│   └── execution.py     execute_cypher (read-only, with a timeout), run_query and Execution (an error as a value)
├── _results/            what comes back; __init__ re-exports everything, so `text2cypher._results` is still the path
│   ├── results.py       Result, Attempt, Records, Status, SchemaScope, total_usage, and the run errors LLMError and DatabaseError
│   └── values.py        GraphNode, GraphRelationship, GraphPath, to_python: what a query returns, without the client
├── skills.py            public: Skill, builtin_skills, load_skills (the names of _skills, for the skills= argument)
├── _skills/             Skill, skills(), load, catalog, check_inline: guides to parts of FalkorDB's Cypher, one folder each (SKILL.md)
├── _errors.py           Text2CypherError, GraphNotFoundError (LLMError and DatabaseError are in _results/results.py, with the attempts they carry)
├── llm/                 public: AsyncChatClient, AsyncLLMClient, Completion, Message, Tool, ToolCall, Usage, OpenAIClient, FakeLLM, Call
│   ├── _base.py  _openai.py  _fake.py  _adapt.py (as_client: the SDK's own client)
├── schema/              public: GraphSchema, Index, Path, SchemaFilter, filter_schema, restrict_schema, render_schema, schema_from_snapshot
│   ├── _model.py  _from_snapshot.py  _filter.py  _stopwords.py  _render.py
├── _validation/
│   ├── validator.py     QueryValidator: the three cypher-guardrail async validators
│   └── issues.py        Issue, Source, ValidationResult, drop_repeated_property_issues
└── _generation/
    ├── prompts.py       generator system prompt (rules built from lists, the checks, the skills section) and the question message
    ├── conversation.py  Round, the turns of a conversation (chat_messages) and the same as one message (folded_message)
    ├── problems.py      the issues of a query worded for the LLM, one handler per issue code; likely_typos
    ├── typos.py         fix_typos: replace a misspelled name in a query with the name it is a typo of
    ├── header.py        repair_header: the CYPHER header the way FalkorDB accepts it
    ├── schema_filter.py LLMSchemaFilter, FilterResult, AsyncSchemaFilter, apply_filter
    ├── read_skill.py    READ_SKILL, chat_reading_skills: the tool loop that lets the model read skills
    ├── extract.py       extract_cypher: first fenced block, else the trimmed reply
    └── generate.py      generate_cypher (a first query or a correction, with or without skills), Generation, ask
tests/                   mirrors src/: llm/ schema/ validation/ generation/ trial/, and test_workflow, test_pipeline,
                         test_config, test_errors, test_execution, test_sample_examples, test_skills, test_examples (the
                         quickstart), test_docs (the links and the index), test_public_api; support.py, conftest.py
examples/                quickstart.py (embedded FalkorDB, scripted LLM, one correction); run by a test
evals/                   the trial harness (not part of the package): dataset (types), datasets/ (movies, company,
                         company_synonyms, skills), replay, judge, metrics, trial, recorded/
docs/                    the Sphinx site (conf.py and the guides) and this document
```

## Components

### Pipeline (`_pipeline.py`)

`Text2Cypher(client, llm, *, model=None, config=None, schema_filter=filter_schema, skills=None).run(question, graph)` (`llm` may be the SDK's `AsyncOpenAI`, with `model`; see LLM):

1. Asks its `SchemaSource` for the graph's `GraphSchema` and the source the validators answer from: a `Snapshot`
   (`types=True`) captured on every call, or, with `Config.schema_refresh_seconds`, a kept one (see Schema), with the
   example values (see below).
2. The source raises `GraphNotFoundError` (a public `ValueError`) before any LLM call if the snapshot has no labels and
   no relationship types: FalkorDB has no graph without data, so an empty snapshot means a missing graph or a typo,
   and nothing is kept for it.
3. Runs the workflow with a `QueryValidator` that answers from the same snapshot (hybrid mode), so the prompt and the
   validator cannot disagree.
4. If `config.execute`, the workflow also runs each query that passed validation (see Execution).

An error from the database while reading the schema propagates, before any LLM call, since nothing has been paid for yet. An exception from the LLM client, or from FalkorDB while a query is validated, ends the run with an `LLMError` or
a `DatabaseError` that keeps the attempts and the tokens so far (see Errors). `clear_schema(graph=None)` forgets what is kept of the schema and the example values.

### Workflow (`_workflow.py`)

`run_workflow(question, schema, *, llm, validate, config, schema_filter=None, execute=None, skills=None)` is the loop. The validator and the executor are
passed as callables, so the loop is tested without a database. State is explicit: the rounds (the failed queries so far, each with its issues or
its execution error and the skills it read), the number of corrections, whether the filtered schema has been given up, and the replies with no text in a row.

- A filtered schema exists only if a `schema_filter` is given (it may be async, `apply_filter` awaits it) and keeps less than the whole schema. A filter that
  keeps everything counts as no filter, and every attempt is `full`.
- A decline (an empty fenced block) on the filtered schema may be the filter's doing, so the same request (the
  first query, or the same correction) is made again with the full schema, once, without using a correction;
  later corrections then use the full schema. Only a decline on the full schema is `unanswerable`. The empty
  attempt stays in `Result.attempts` with its usage.
- A reply with **no text at all** (empty or only white space) is not a decline: `_no_query` records an attempt with `empty_reply`, and the loop asks the same request again once, exactly as it was (the
  rounds, the schema and the scope do not change, and no correction is spent). A second one in a row ends the run as `failed` with no query. Any reply that has text resets the count. ADR-047. The reason the reply ended decides first, when the client says (ADR-048): a `refusal` has no empty flag and is a decline; `length` and `content_filter` (`_NOT_RETRIED`) end the run at the first empty reply, since the same request would hit the same limit or filter.
- An empty query is never validated: the validator would call it a syntax error and the loop would spend its whole
  budget on it.
- `Config` rejects negative values and `total_attempts < attempts_w_filtered`.

### Results (`_results/`)

`Result` has `status`, the last `query`, every `Attempt`, `usage`, `records`, `execution_error` and `filter_usage`. An `Attempt` has the query (empty for a decline or an empty reply), its `ValidationResult` (`None` for no query, and
for a query the database failed to validate), the schema scope, `Usage`, `execution_error`, and the flags `auto_fixed`, `empty_reply` and `skills` (the ids read) and `finish_reason` (why the reply ended). `Result.usage` is the sum of the attempts' usage and
`filter_usage` (`NO_TOKENS`, a zero, for a filter that asks no model), or `None` if any of them did not report it (`total_usage`, which the run errors use too). `Issue`, `ValidationResult` and `Attempt` are hashable (the dict inside
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
  defects here (`ParameterNotProvided`: the query runs as it is, with only the parameters of its own `CYPHER` header; `UnknownIndex`: the query would return
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
  `paths` (a tuple of one-hop `Path(source, type, target)`), `indexes` (a tuple of `Index(entity, property, type)`,
  with `type` `RANGE`, `FULLTEXT` or `VECTOR`, read from the snapshot; the filter keeps those of what it keeps), `functions` (the instance's user-defined
  functions as `library.function`, which belong to the instance, so the filter keeps them all) and `node_examples` and `relationship_examples` (see Example values).
- `schema_from_snapshot(snapshot)` builds it from a `cypher-guardrail` `Snapshot` captured with `types=True`.
  FalkorDB has no declared schema, so the snapshot defines it by the data: names, properties with value types,
  and edges as label-set triples. Each edge expands into one `Path` per label pair (a multi-label node contributes
  a path per label). Value types need a full scan (`types=True` cannot be combined with `sample=`), so nothing is
  sampled. A snapshot captured without types is rejected (`ValueError`); a graph with no data gives an empty schema.
  The snapshot also feeds the validators (cypher-guardrail's hybrid mode): they answer lookups from it and ask the
  database only to plan the query (`EXPLAIN`) and to confirm a problem before reporting it. Checked on the test graph:
  identical issues and scores for every kind of issue, and a query that is fine costs one request instead of up to ten.
- **Clearing what is kept.** `Text2Cypher.clear_schema(graph=None)` calls `SchemaSource.forget`, which drops the live snapshot and the example values of one graph or of all, so the next run reads
  them again. It is synchronous and queries nothing. The explicit clear is what another project exposes as an endpoint; here it also covers the examples, which a changed value inside the
  same structure would otherwise never refresh.
- **Keeping the schema.** `Config.schema_refresh_seconds=None` (the default) captures on every run. A number keeps one
  `AsyncLiveSnapshot` per graph: when the time is up, `Text2Cypher` awaits its `refresh()`, which compares a fingerprint
  (names, indexes, node and relationship counts) and scans again only if it differs. The live snapshot's own background
  refresh is switched off (it is never due), so no task outlives the event loop. A graph that did not exist is not kept,
  so one created after the error is found. The fingerprint is a heuristic, and between checks a change is missing from the
  prompt; it is not missing from validation, because a problem is confirmed against the database before it is reported.
- **Example values** (`Config.example_values`, default 3; `0` turns them off). `GraphSchema.node_examples` and
  `relationship_examples` map entity then property to values already written as in a query (`'Alien'`, `1999`,
  `true`; a string is cut at 40 characters, with an escaped quote), and `render_schema` prints them after the type,
  `title: STRING (e.g. 'Alien', 'Heat')`. When any exist the prompt gets one rule: "Where the schema shows example
  values, write a value the way the examples are written (spelling and case)". They are read by `_graph/examples.py`: one
  read-only query per property of a plain type (String, Integer, Float, Boolean; not a list, map, point or vector)
  that returns the first 20 non-null values and stops (a `DISTINCT` would read every row), of which the first
  three distinct are kept. At most 8 run at once, each with a 2 second timeout, and a query that fails or times out
  gives that property no examples, because they are a help and not a need. The filter keeps the examples of what it keeps.
  They are read once per structure of the graph: `SchemaSource` keeps them with the snapshot they were read for (compared
  without its capture time) and reads again only when a new snapshot differs in structure, so a default `Text2Cypher`
  does not repeat the queries on every run (checked: a second read costs what a read with no examples costs). Values that
  change inside the same structure are not seen, which is fine for illustrations. The option is on by default because
  how a value is written (`'Sales'` or `'sales'`) is what a correct query most often depends on, and it is an option because
  they are data from the graph that goes to the LLM provider.
- Names containing a backtick are skipped: FalkorDB has no escape for one, so no query can use such a name, and
  skipping them also removes any chance of injecting through a name read from the database. Other names are quoted
  only through `quote_identifier`, and only when rendered.
- `render_schema` produces the Neo4j-style text the generator prompt carries: `Node properties:`,
  `Relationship properties:`, `The relationships:`, and `Indexes:` and `User-defined functions:` when there are any. Types are upper-cased and joined with `|`; names that are not
  plain identifiers are backtick-quoted; sections with nothing to say are left out.
- **Filtering** (`filter_schema(schema, question)`, the default `schema_filter`) matches the question's words to
  the words of label and relationship-type names: Snowball stems (`snowballstemmer`), camelCase and underscore
  splitting, case-insensitive. Stopwords are left out of names (so "in" cannot find `IN_GENRE`) unless a name is
  only stopwords. A matched label keeps the paths touching it, a matched type keeps its paths, and every label and
  type on a kept path stays with its properties, so the LLM can use the far end. Property names are not matched. A
  question that matches nothing keeps the whole schema, so a miss costs nothing; the lexical filter does not know
  that "people" means `Person`. A `SchemaFilter` is a plain synchronous callable `(schema, question) -> schema`, and `restrict_schema(schema, labels, types)` is what it ends with (public, so a filter of
  your own can too): the named entities, the paths that touch them, the entities on those paths, and their properties, indexes and examples.
- **The LLM filter** (`LLMSchemaFilter(llm=None, *, min_entities=20, model=None)`, in `_generation/schema_filter.py` because it needs the LLM protocol and `schema` and `llm` are sibling layers). Choosing it replaces the
  lexical filter: it runs alone and is shown the whole schema, so there is no pre-run to cut what it sees. The prompt: a system message that asks for the labels and relationship types a query would use, only names in the
  schema, the question as text and never instructions, and a reply of JSON in one fenced block; a user message with the labels and their property names (no types, examples or indexes), the paths and the question.
  The reply is read from a fenced block or the first `{...}`, names are matched without case and unknown ones dropped, and `restrict_schema` does the rest. An unusable reply (no JSON, no known name, empty lists), a
  call that raises, or a schema below `min_entities` (labels plus the types on paths) keeps the whole schema, so it can only cost tokens, never a question; a failed call is not counted, because nothing says what it cost.
  Cancellation passes through. With `llm=None` the `Text2Cypher` makes a copy (`bound`) that has the main client; `model=` asks the main client with another, cheaper model (`OpenAIClient.with_model`), or names the model of an SDK client the filter is given. It is a frozen dataclass, safe to share.
- **Async filters and what they cost.** `Text2Cypher(schema_filter=...)` takes a sync function, an async one, or an `AsyncSchemaFilter`; `apply_filter` calls it, awaits it if it is awaitable, and wraps a bare
  `GraphSchema` as `FilterResult(schema, Usage(0, 0))`. `FilterResult.usage` is zero when no model was asked and `None` when one was and did not report. It goes on `Result.filter_usage` (zero by default) and
  into `Result.usage`, and into the `usage` of `LLMError` and `DatabaseError`, since it was paid for before the first attempt. `NO_TOKENS = Usage(0, 0)` is the zero the library uses for "asked no model", as for a fixed
  typo, where `None` means unknown.

### LLM (`llm/`)

- Two stateless protocols. `AsyncChatClient` has `chat(system, messages, tools=None, *, tool_choice="auto") -> Completion(text, usage, tool_calls)`, with
  `messages: Sequence[Message]` (`Message(role, content, tool_calls, tool_call_id)`, roles `user`, `assistant` and `tool`, the first from the user) and
  `usage: Usage | None`; the library calls it when a client has a callable `chat`. `AsyncLLMClient` keeps its one method,
  `complete(system, user)`: a client with only that is given the same conversation as one user message. Changing the signature of `chat` is a deliberate change of a public protocol, so it is
  a major version (2.0.0) with a `MIGRATION.md`; at run time a client without the new arguments keeps working, because tools are offered only to a client that takes them. A client with both gets `chat`; an attribute named `chat`
  that is not callable (an SDK client's namespace) does not count. The choice is made in `generate_cypher`, not in an
  adapter, because the one-message form lists the attempts and ends with the instruction once, which a generic
  adapter over turns could not do without parsing the prompt text.
- **Tools** (`Tool`, `ToolCall`), shaped like the common denominator of the OpenAI SDK, MCP, Anthropic and LangChain: a tool is a name, a description and a JSON Schema for its parameters; a call is an id, a name
  and arguments as a **dict** (OpenAI sends a JSON string, which `OpenAIClient` parses; invalid JSON gives `{}` so that the answer can say what is missing); an answer is a `"tool"` message carrying the call's id. The client
  maps them to the API: tools as `{"type": "function", "function": {name, description, parameters}}`, an assistant turn with calls as `tool_calls` with the arguments as a JSON string, a tool turn as
  `{"role": "tool", "tool_call_id", "content"}`; every call id gets an answer, as OpenAI requires; calls of a kind other than a function are ignored; `tool_choice` is sent only with tools, and `"none"`
  (not removing the tools, which some providers refuse when the history has tool turns) is how a text answer is forced. `supports_tools(llm)` says whether a client can be offered tools: its
  `supports_tools` attribute if it has one, else whether `chat` takes `tools` or `**kwargs` (a client that takes `**kwargs` and ignores the tools just returns text, which the loop treats as the answer).
  `OpenAIClient(tools=False)` and `FakeLLM(tools=False)` say they cannot. `FakeLLM` records the tools and the choice on each `Call` and scripts a tool call with a `Completion` that has `tool_calls`.
- **The SDK's own client is accepted** wherever an LLM is (`llm/_adapt.py`, `as_client(llm, model)`): an `openai.AsyncOpenAI` (also `AsyncAzureOpenAI`, and one for a compatible
  server) with `model=` becomes an `OpenAIClient` with its defaults. `model=` is required for it (the SDK client has none) and an error with any other kind of client, so it is never silently
  ignored; the synchronous `openai.OpenAI` is a `TypeError`, because the library is async and running it in a thread would break that. `Text2Cypher(client, llm, *, model=None, ...)` and
  `LLMSchemaFilter(llm=None, *, min_entities=20, model=None)` take it; for the filter, `llm=None` plus `model` asks the main client with another model (`OpenAIClient.with_model`, a copy that
  shares the SDK client and every setting), which needs an OpenAI client as the main one. `OpenAIClient` keeps its name and is the full-control form: the options live there.
- `OpenAIClient(model, *, client=None, temperature=0, system_role=True, timeout=120.0, max_retries=None, tools=True)` uses `AsyncOpenAI` (Chat Completions); pass your own client
  for an OpenAI-compatible server, and `temperature=None` for a model that only accepts its own. The SDK's default timeout is 10 minutes and a question makes up to a dozen calls, so
  the client sets 2 minutes through `with_options` (a copy that shares the connection pool, so it also applies to a client you pass; `None` leaves that client's own). Retries are the
  SDK's (2 by default; the option changes them); a call that still fails ends the run with an `LLMError`. `FakeLLM` replays
  scripted replies and records its calls, for tests.

### Generation (`_generation/`)

- **Extraction**: the first fenced block anywhere in the reply, else the whole reply trimmed (models add prose around
  fences, and validation catches what is left). A language tag (any word on the fence's own line, or `cypher`,
  `opencypher`, `cql` on a one-line block) is dropped; Windows line endings are accepted. An empty block means the
  LLM declined. A reply with no text at all is not a decline: `Generation.empty` is set, and the loop asks again once unless the reason it ended says not to (see Workflow, ADR-047, ADR-048). `Completion.finish_reason` is carried to `Generation` and to `Attempt`. A reply with no fence at all is taken as the query: a prose refusal then fails validation and the
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
  no correction. About 55 words in every system prompt. With `Config.use_parameters` it also checks
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

### Typos and the header (`_generation/typos.py`, `_generation/header.py`)

- `Config.autofix_typos` (off). After a query fails validation, `fix_typos(query, issues, schema)` replaces each label,
  relationship type and property that the validators report as unknown (`UnknownLabel`, `UnknownRelationshipType`,
  `UnknownPropertyKey`) by the one name of the graph it is a likely typo of. Such a name exists nowhere in the graph, so
  every use of it is wrong and each is replaced; nothing needs to know which variable it belongs to. A property that
  exists on another label (`InvalidPropertyAccess`) is not touched, because the variable it belongs to is not known.
- The rule for "a likely typo" is the one behind the "did you mean" (`likely_typos` in `problems.py`: very similar, or
  fairly similar and starting alike; `Client` is not `City`), and a name is fixed only when **exactly one** name of the
  schema passes it, so an ambiguous one (`Persn` with `Person` and `Persons`) is left. Candidates come from the full
  schema, because the query is validated against the whole graph. A case difference counts (`person` is `Person`).
- The rewrite works on the code of the query only: strings, line comments and block comments are cut out first and put
  back. A label or type follows a colon (or a bar: `[:A|B]`), a property follows a dot (not before an open
  parenthesis) or is a key before a colon, and backticks around the old name go with it. The new name is quoted only when
  it needs it.
- The fixed query is validated again, and fixing is tried up to three times, because a wrong label stops the
  validators from checking its properties: they show only once the labels are right (checked on FalkorDB: a query with
  `Persn`, `ACTD_IN`, `Movi`, `nme` and `titel` needs two). It is kept only if it ends up passing. If it does not,
  it is dropped and the LLM gets the query it wrote, with the problems it had, as without the option, so a fix can only save a
  call.
- A fix is an `Attempt` of its own after the query the LLM wrote (`auto_fixed=True`, `usage=Usage(0, 0)`: a zero,
  because `None` would make `Result.usage` unknown), so both the original and the fixed query are on the result. It is
  not a request, so it is free: it does not count against `total_attempts`, and a query can be valid with
  `total_attempts=0`. The fixed query is the one that is executed, if execution is on.
- Off by default because a fix can make a query valid and still not what the question meant (a typo of `Person` that was
  meant to be `Persona`, or a property of the wrong label that happens to be right on another).
- **The parameter header** (`repair_header`, `use_parameters` only, no option): FalkorDB rejects `CYPHER a='x', b=2` (commas) and a header without `CYPHER`, each a
  `SyntaxError` that hides every other problem and used to cost a correction (checked on the real server; a lowercase `cypher` and a newline between pairs are accepted). It adds a missing `CYPHER` and writes
  the pairs with one space between, reading each value as a string, a list, a map or a word, so a comma inside a string or a list is kept; a value that is never closed, or no header, gives
  nothing. It changes nothing about what the query means, which is why it needs no option and is not like the typos. It shares the typos' path (`_check`): validate, repair, validate again (up to three
  rounds, the header first), kept only if it passes, as an `Attempt` of its own with `auto_fixed=True` and `Usage(0, 0)`.

### Errors

- One base, `Text2CypherError`; `GraphNotFoundError(Text2CypherError, ValueError)` keeps its old base. The two errors of a run that stops in
  the middle are `LLMError` and `DatabaseError`, subclasses of a private `RunInterrupted` that holds `question`, `attempts` and `usage`
  (computed by the `total_usage` helper that `Result.usage` also uses, so they cannot disagree). They live in `_results/results.py` and not in
  `_errors.py` because they carry `Attempt`, and `_errors` is the bottom layer.
- Why: the LLM calls are paid for, and an exception that lost them did so silently. Execution already kept them (`execution_error`); this does the same for
  the other two things that can fail in the loop. The failed call's own cost is unknown, so it is not counted, and the docs say so.
- `LLMError` wraps any `Exception` the client raises (provider errors are arbitrary), with the original as `__cause__`; `CancelledError` is a
  `BaseException` and passes through. This changes what a caller sees: code that caught the provider's exception must read `__cause__`, so it is in the changelog under
  Changed. `DatabaseError` wraps only `redis.exceptions.RedisError` from validation (a programming error is a bug and propagates), and the query that
  was being validated is kept as an attempt without a validation. The schema read, before any LLM call, loses nothing and is not wrapped.
- `RunInterrupted` defines `__reduce__`, so the errors survive a pickle (a process pool, a task queue) with their attempts; `__cause__` does not, as for any exception.
- `FakeLLM` raises a reply that is an exception, so this can be tested.

### Skills (`_skills/`)

- A skill is a folder with a `SKILL.md` (the name the agent-skills convention uses; the loader finds it whatever the case, listing the folder, so a Mac and Linux agree): a front matter (`name`, a one-line `description`) and a guide whose examples are in `cypher` blocks. The folder's name is the id. `skills()` loads every folder, sorted, once;
  adding a skill is adding a folder. They ship in the wheel and the sdist, and `THIRD_PARTY_NOTICES` (the MIT notice for the path-finding skill, adapted from FalkorDB's own) ships as a license file.
- Three skills, for the large and rarely needed: `graph-algorithms` (`algo.pageRank`, `betweenness`, `WCC`, `labelPropagation`, `BFS`), `path-finding` (`algo.SPpaths` and `algo.SSpaths` with their options, variable-length
  patterns, where `shortestPath` and `allShortestPaths` work) and `functions` (what FalkorDB names differently or lacks, and the string, list, math, date, place, vector and aggregate functions it has). The inline rules of the prompt stay:
  they are short, common and already checked.
- **Every example is run.** The tests run each `cypher` block of each skill with `ro_query` on a graph of people and cities and require rows, so a skill that teaches a write or a wrong signature fails the build; the claims that a
  syntax or function is missing (`!=`, `=~`, `COUNT {}`, `EXISTS {}`, `datetime()`, `toLowerCase`, `apoc`, `shortestPath` in `MATCH`) are each asserted to fail on the server, so a FalkorDB that adds one tells us to change the text.
  Each skill is also capped at 3000 characters, because reading one costs tokens.
- Facts found while writing them, which a model would not know and one skill disagreed with: `algo.pageRank` takes the label and the type as two strings, not a map; `algo.BFS` takes the start node, the depth (`-1` for no limit) and the type (`null` for any),
  and its `nodes` leave out the start; `algo.MSF` yields `edges` and `nodes`; `algo.SSpaths` returns up to `pathCount` best paths from the source across all destinations (not one per destination; `0` is those that tie for best);
  `allShortestPaths` works only in `MATCH` and `shortestPath` only in `WITH` or `RETURN`.

- **Skills you provide (ADR-050).** `Text2Cypher(skills=[...])` replaces the shipped ones; `builtin_skills()` and `load_skills(path)` (in `text2cypher.skills`, with `Skill`) make a list to extend or replace them. The catalog is one `dict[str, Skill]` held by the pipeline and passed down (`run_workflow(skills=)`, `generate_cypher(catalog=)`, `SkillsOffer.catalog`, `chat_reading_skills(catalog)`); `None` means the shipped ones, which keeps every earlier call as it was. `Skill.__post_init__` checks the id, the one-line description and that no `cypher` example writes (a clause not preceded by a `.` or a word character, with strings removed, so `list.remove` is a function); `catalog()` rejects an empty list and a repeated id; `check_inline` caps the total at `INLINE_LIMIT` = 16,000 characters for a client without tools. `skills=` without `use_skills` is an error, not ignored. Not checked: that a user's examples run (only the shipped ones are tested on a server) or that its claims are true. A skill is text sent to the provider that steers the model, which the docs say.

### Reading skills (`_generation/read_skill.py`)

- `Config.use_skills` (off). The six prompt parts were approved one by one: a rule line when tools are offered ("Skills are short guides to parts of FalkorDB that are easy to get wrong. Read a skill, with read_skill, only when the question
  needs what it covers, before you write the query."); a list after the rules, `- id: description` for each skill not yet read; "Skills you have already read:" and their text; the tool `read_skill(id)` ("Read one skill: what it is for, the syntax, and
  an example that runs on FalkorDB."); its answers (`# name` and the guide, or "There is no skill with the id `x`. The ids are: ..."); and, for a client that cannot be offered tools, "Skills (guides to parts of FalkorDB that are easy to get wrong):" with every
  skill's text and neither a rule nor a tool. The section sits after the rules and before the checks; the default prompt, without the option, is byte for byte what it was.
- **The loop** (`chat_reading_skills`, per request): up to three rounds in which the model's calls are answered (every call id gets an answer, as OpenAI requires; a fifth call in one round, and any after, is answered "Too many skills at once: ask for fewer." and not read),
  then a last call with `tool_choice="none"`, the tool still offered. A missing or non-string id is told to give one, and an unknown id is told the ids. The tokens of every call are summed (unknown if any is) into the attempt's `usage`, and the ids read, in order and without repeats,
  are `Generation.skills` and `Attempt.skills`. If a call fails after others succeeded, `SkillLoopFailed` carries what they cost and `_write` turns it into an `LLMError` whose last attempt has no query and holds that usage.
- **Remembering.** A `Round` has `skills_read`; the next request's prompt gets the union of its rounds' skills as "already read" and lists only the rest, so a correction does not read them again and the conversation stays plain turns (no tool turns are replayed). A decline's reading is not
  kept (the declined request is asked again, and reads again).
- `supports_tools(llm)` decides: a client that can be offered tools lists and reads; any other (`chat` without tools, or `complete` only) has every skill in the prompt, about 2,000 tokens a request and no extra call.

### Execution (`_graph/execution.py`)

- Off by default (`Config.execute = False`).
- `execute_cypher(graph, query, *, timeout_ms=None) -> Records` runs `ro_query` with a timeout: 10 s by default
  (`Config.execute_timeout_ms`), `None` for no limit, because a generated query can be expensive and an unlimited
  default would let one hang the caller.
- Execution is part of the loop (`run_workflow(..., execute=...)`): a query that passed validation is run at once.
  `run_query(graph, query, *, timeout_ms)` returns an `Execution(records, error, correctable)`: any
  `redis.exceptions.RedisError` becomes an `error` string and anything else is a bug and propagates.
  `Result.records` is `Records(columns, rows)`, with the client's own values (`Node`, `Edge`, `Path`). `Records.as_python()` (ADR-052) gives the same rows with them as
  `GraphNode`, `GraphRelationship` and `GraphPath` (`_results/values.py`): frozen, read-only properties (`MappingProxyType`), equal and hashed by id, picklable (`__reduce__`, since a mapping proxy is not), converted inside lists, tuples and
  maps (a map comes back as a `dict`), idempotent. `rows` stays raw, so nothing changed for anyone and the conversion costs nothing unless asked for. No ORM is imported; a user builds entities from `labels` and `properties`.
- `correctable` is true for a `ResponseError` that is not a timeout (FalkorDB rejected or failed this query: a type
  error on a value, a division by zero, a bad argument). Checked on FalkorDB with queries that pass both validators
  (`substring(name, -1)`, `year % 0`, `name / 2`). Its message becomes the next turn of the conversation
  ("FalkorDB accepted this query but it failed when it ran: ...", the error unchanged, then an instruction ending
  "fixes this and keeps what already works"), costs a correction from `total_attempts`, and is on the attempt as
  `execution_error`. When the corrections run out on such queries the status is `failed` (no query both validated
  and ran), with `Result.execution_error` the last error; before, it stayed `valid`.
- A timeout (also a `ResponseError`, recognised by "timed out", which a test pins on the real server) and a lost
  connection are not correctable: they say nothing about the query, and a rewrite would not help. The loop ends with
  `valid`, the query and `Result.execution_error`, so the LLM calls are never lost.
- An empty result is not an error and is never corrected. FalkorDB's error text can quote a value from the data and
  goes to the LLM, which the docs say. There is no cap on returned rows: the timeout bounds time, not memory, and a
  result-size limit belongs on the server.

## Tooling and constraints

- Python `>=3.12` (CI matrix 3.12, 3.13, 3.14): the embedded FalkorDB used by the tests needs 3.12, and nothing needs
  a newer feature. `py.typed` is shipped; the classifiers say production/stable, Python 3.12-3.14, typed.
- uv for everything (`uv add`, `uv run`); ruff for lint and format (rules B, I, UP, SIM, C4, C90 with a
  complexity limit of 10, PTH, RET, RUF, ASYNC, PT, TID, TC, FURB); ty for types; import-linter for the layers above; pytest (importlib import mode, strict markers); coverage with branches. See
  `CLAUDE.md`.
- Dependencies: `cypher-guardrail>=1.5.1`, `falkordb`, `openai`, `redis` (for `RedisError`), `snowballstemmer`.
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
  `reply`, `make_issue`, `invalid`, `misspelled` and `VALID`, `ScriptedValidator` and `ScriptedExecutor`, a small hand-built schema, and `openai_client`,
  `chat_reply` and `tool_calls_reply`, which build the real OpenAI SDK over a mock transport that can fail, time out and reply in turn).
- Every mutation check of a new behaviour is done from a copy of the file and restored from it, never with `git checkout`, which also removes uncommitted work.
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
  never match a name ("people" for `Person`), which fell back to the full schema.
- The measurement of the LLM filter and of skills (2026-10-21, ADR-049): the replay LLM answers `chat` (a `CALL <tool> <JSON>` reply stands in for a tool call), a run takes a `Condition` (filter, skills, example values), and two datasets
  were added: `company_synonyms` (20 questions on the 36-label schema, worded unlike it, judged by validation) and `skills` (15 questions on a small graph with data, judged by rows). Haiku and Sonnet agents, one run per cell: the LLM
  filter was right on the first schema for 20 and 19 of 20 questions against 12 and 11 for the lexical filter (every question ended valid either way, since a miss is retried on the full schema); skills lifted Haiku from 10 to 12 of 15 and made
  no difference for Sonnet (15 of 15 both ways). The replies are in `evals/recorded/measurement/` and `tests/trial/test_measurement.py` pins the numbers. The defaults stay off. Limits: one run per cell,
  agents and not the API, validation-only judging for the company schema.

## Not built yet

Each item says what would bring it back.


**Ideas**
- Telling the model, in a correction, that its last query was cut off (`finish_reason` is `length` on an attempt that has a query); today the query is validated and corrected like any other.
- Reversing a wrong-direction path without the LLM: the validator gives the exact replacement. Wait for a sign that models get directions wrong often.
- Feeding an empty result back to the LLM, opt-in: an empty result can be the right answer; a confidence like another project's would be a different design, and needs an answer-writing step.
- A cap on the number of sampled properties of a very wide schema, and sampling only indexed or low-cardinality ones (ADR-034).
- More skills, rewritten after running their claims on the server (skills of the user's own are built: ADR-050).
- Query optimization from cypher-guardrail's `Advisor` (a limit, a bound on a path, a label, an index): looked at and dropped (ADR-051), because every suggestion that rewrites a query changes what it returns. Index advice for the owner of the graph, as a report over many runs, would bring it back.
- Adapters for other LLM apps (Anthropic, LangChain) and an `OpenAIResponsesClient`: the tool shapes map one to one (ADR-042).
- Whether structured issues can inform escalation: an unknown label that exists nowhere in the graph will not be fixed by the full schema. Untested.
- A sync facade, if demand appears (see Async only). `run_many(questions, concurrency=N)` for dataset evaluation.

**Left out of the prompt, each until evaluation or a feature needs it**
- `toLower()` guidance beyond the index rule: a function on an indexed property prevents index use, but a case-insensitive match is often what the question means. Decide with the harness.
- `LIMIT` guidance (adding a limit the question did not ask for changes the answer).
- Few-shot examples generated from the schema; the schema as inline Cypher patterns instead of the Neo4j-style text; the schema in the user message instead of the system prompt (ours keeps a stable prefix for provider
  prompt caching); a numbered `code: description` error format for the correction turns.
- Answer synthesis (a written answer from the rows).

**Other**
- Embedding similarity for the lexical filter (a heavy dependency, would be an extra) and property-name matching: when the filter's miss rate justifies them.
- A row limit; converting rows to plain Python.
- Metrics for the harness: success rate, self-healing rate (failures fixed by a correction), validation failures by type.
