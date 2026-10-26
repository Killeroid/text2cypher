# Recorded replies

The replies of the second trial (2026-10-20), written by Claude Haiku agents that were given the prompts
`text2cypher` produced at the time, one agent per call, and saved as `<question id>_<n>.txt`.

- `base/`: the 24 basic and harder questions (`evals/datasets/movies.py`), plus the second reply of the three
  questions that were declined on the filtered schema and asked again on the full one.
- `seeded/`: the 9 seeded questions (the first reply is the flawed query written by `evals.trial seed`,
  the next ones are the agents' corrections).

`measurement/` holds the replies of the measurement of the LLM schema filter and of skills (2026-10-21), one folder
per cell, from Haiku and Sonnet agents (see `evals/README.md`); `tests/trial/test_measurement.py` pins their numbers.

The others are the answers of one small model on one day, kept as a regression fixture: replaying them through the
current code must give the outcomes pinned in `tests/trial/test_recorded_trial.py`. That pins the behaviour of the
schema filter, the validator, the issue merging, the loop (corrections, the retry on the full schema, giving up,
declining) and the execution. It does not pin the wording of the prompts (a reply is not compared with the prompt it
answered); `tests/generation/test_prompts.py` and `tests/generation/test_prompt_rules.py` cover that. Record new replies (see
`evals/README.md`) when the prompts change on purpose and the old replies no longer fit.

`base/q07_1.txt` was edited on 2026-10-26: the agent's query had no `DISTINCT`, and FalkorDB 4.22 returned two rows for it where 6.0 returns the correct four (two movies by two directors), so the recorded outcome had come from a defect of the older server. The reply now has the `DISTINCT` that the question's reference query has.
