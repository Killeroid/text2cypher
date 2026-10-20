# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

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
