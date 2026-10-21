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

`large.py` is a company graph with 36 labels, 44 relationship types and 46 paths, written as a
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

## Code

`data.py` (the film graph and questions), `replay.py` (runs the questions with the replay LLM),
`judge.py` (the verdicts), `metrics.py` (the numbers), `trial.py` (the files, the report and the
command line), `large.py` (the large schema).

## What it does and does not show

It exercises the prompts and the loop with real replies. When a Claude agent writes the
replies it is a stand-in for a model, not a bare API call: it has tools and context of
its own and reads the system and user text as one document. One small model, one pass
and a small graph say little about how the prompts do in general, and about how much the
filter prunes on a large schema. A trial of 37 agents took about a million tokens.

`tests/trial/test_trial.py` checks the harness itself with the gold queries.
