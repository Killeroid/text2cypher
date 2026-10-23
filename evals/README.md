# Trial harness

A way to try `text2cypher` against a model, or a person, without an API key, and to
see what the library does with the replies. It is not part of the package.

## How it works

`evals/trial.py` runs every question through the real `Text2Cypher.run` (schema
filter, validator, corrector, execution) with a *replay* LLM. The LLM is whoever
writes `DIR/<question id>_<n>.txt`: reply `n` to that question's `n`-th call. When a
question needs a reply that is not there yet, its prompt is written to
`DIR/pending/<id>_<n>.txt`. Write the reply next to the others and run `round` again.

```
uv run python -m evals.trial round evals/work/run1             # first replies are missing: prompts appear in pending/
uv run python -m evals.trial seed  evals/work/run2             # flawed first queries for the seeded questions
uv run python -m evals.trial round evals/work/run2 --seeded    # the corrector prompts appear in pending/
uv run python -m evals.trial gold  evals/work/gold             # the gold query as every first reply (a self-check)
```

`evals/work/` is git-ignored. Each `round` starts a throwaway embedded FalkorDB server
with the trial graph (so it needs the dev dependencies) and prints one line per
question: the status, how many attempts and with which schema (`f` filtered, `F` full),
how much of the schema the filter kept (and whether the gold query needed a name it
dropped), and the verdict.

## Questions

`evals/datasets/movies.py` has the graph (6 labels, 8 relationship types), 24 questions with a
gold query or the expectation that the LLM declines, and 9 *seeded* questions: a flawed
first query, so that the corrector prompt and the corrections can be tried.

A question is judged by running the final query and the gold query and comparing the
rows by their values, not their order; a duplicate row counts, so a reply without the
gold query's `DISTINCT` is a mismatch. Extra columns are accepted (a rating next to the
title that was asked for) when every gold row has a returned row of its own; missing
columns, other values or another number of rows are a mismatch. The note on the filter
says how much of the schema it kept and flags a label or relationship type that the gold
query names and the filter dropped.

`gold` and `seed` refuse a directory that already has files, so replies of an earlier run
are never replayed by mistake; use a new directory for new replies. Files are UTF-8.

When every question has finished, the report ends with the numbers of `metrics.py`: answered
correctly, right on the first query, the share of bad first queries healed by correction (and the
corrections each took), declined as expected, and how often each validation issue code was
reported. The recorded trials pin them in `tests/trial/test_recorded_trial.py`.

## A large schema, without a model

`datasets/company.py` is a company graph with 36 labels, 44 relationship types and 46 paths, written as a
snapshot (a schema needs no data), with 16 questions and their gold queries. `check_gold` validates
each gold query against it, offline, and `measure` runs the schema filter on each question: how
many of the 80 names it kept, and which names the gold query needed and it dropped. It needs no
model and no database; `tests/trial/test_company_schema.py` pins the result.

## Measuring the LLM schema filter and skills

`--filter {lexical,llm,none}`, `--skills` and `--example-values N` run the questions under a `Condition`;
`--only q1,q2` runs some of them. A reply of `CALL read_skill {"id": "..."}` lines stands in for a native tool call.
`datasets/company_synonyms.py` (20 reworded questions on the 36-label company schema) is for the filter,
`datasets/skills.py` (15 questions on a small graph with data) for the skills. One agent per cell: an agent that
handled two cells mixed their replies once. The replies of the cells that were measured are in `recorded/measurement/`,
and `tests/trial/test_measurement.py` replays them:

| | Haiku | Sonnet |
|---|---|---|
| lexical filter, right on the first schema | 12/20 | 11/20 |
| LLM filter, right on the first schema | 20/20 | 19/20 |
| skills off, answered correctly | 10/15 | 15/15 |
| skills on, answered correctly | 12/15 | 15/15 |

Every company question ended valid either way, since a miss is retried on the full schema; the company graph has no
data, so those are judged by validation alone. One run per cell by an agent, not a bare API call: read it as a signal.
Not measured: example values, the check section, a larger schema, other models.

## Measuring the format of the prompts (ADR-055)

`format_trial.py` compares three layouts of the same prompts, **plain** (the package's own until the Markdown layout of 2026-10-23, rebuilt here), **Markdown** (the package's own now, adopted before this was measured and to be reverted if it does worse)
and **XML**, with a real model through an OpenAI-compatible endpoint, with one call per question. The
layouts are in `formats.py` (plain and XML are built from the package's own rules, checks, examples and schema
sections, and Markdown is the package's prompt, so only the layout differs); `dictionaries.py` has a data dictionary for the movies and the company
graphs (ADR-054); `live.py` loads the settings and makes the clients.

The settings are in `.env` at the root (git-ignored): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHEAP_MODEL`,
`EXPENSIVE_MODEL` and `EXPENSIVE_MODEL_REASONING_EFFORT`, one `NAME: value` (or `NAME=value`) to a line. Nothing
prints a setting. The cheap model is asked with temperature 0; the expensive one with its reasoning effort and no
temperature.

```
uv run python -m evals.format_trial prompts            # write the prompts to evals/work/formats/prompts/ to read
uv run python -m evals.format_trial run --dry-run      # the whole matrix with a stand-in model (no key, no network)
uv run python -m evals.format_trial smoke              # one call to each model: do the settings work?
uv run python -m evals.format_trial run --limit 3      # the real thing, 3 questions a cell (a cheap first look)
uv run python -m evals.format_trial run                # the real thing
uv run python -m evals.format_trial report             # the tables, from what is saved
```

Two experiments. **generator**: the whole pipeline with no corrections and the full schema (the format acts on
the first query), the dictionary off or on, on `movies` (rows compared) and `company_synonyms` (validation only).
**filter**: `LLMSchemaFilter` alone on the company schema, with no dictionary, with the descriptions of labels and
relationship types (B), or with those of properties too (C), judged by whether it kept every name the gold query
needs. 42 cells in all, about 900 calls. A cell is saved when done (`evals/work/formats/<cell>.jsonl`, with the
query, the verdict and the tokens of each question) and is not run again unless `--force`, so an interrupted run
resumes. A call that fails is a row with an `error`, counted apart and never as a success (the filter keeps the
whole schema when its call fails, which would look right).

Read the results as a signal: one run per cell at temperature 0, 24 and 20 questions, so a difference of one or two
questions is noise. The dictionaries describe what the graphs are in plain words, and `company_synonyms` was written
to need exactly that, so a gain from the dictionary there is the most it can give.

## Code

`dataset.py` (the types), `datasets/` (`movies.py`, the film graph and questions; `company.py`, the large schema;
`company_synonyms.py` and `skills.py`, for the measurement), `replay.py` (runs the questions with the replay LLM),
`judge.py` (the verdicts), `metrics.py` (the numbers), `trial.py` (the files, the report and the
command line).

## What it does and does not show

It exercises the prompts and the loop with real replies. When a Claude agent writes the
replies it is a stand-in for a model, not a bare API call: it has tools and context of
its own and reads the system and user text as one document. One small model, one pass
and a small graph say little about how the prompts do in general, and about how much the
filter prunes on a large schema. A trial of 37 agents took about a million tokens.

`tests/trial/test_trial.py` checks the harness itself with the gold queries.
