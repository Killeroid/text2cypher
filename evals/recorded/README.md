# Recorded replies

The replies of the second trial (2026-10-20), written by Claude Haiku agents that were given the prompts
`text2cypher` produced at the time, one agent per call, and saved as `<question id>_<n>.txt`.

- `base/`: the 24 basic and harder questions (`evals/datasets/movies.py`), plus the second reply of the three
  questions that were declined on the filtered schema and asked again on the full one.
- `seeded/`: the 9 seeded questions (the first reply is the flawed query written by `evals.trial seed`,
  the next ones are the agents' corrections).

They are the answers of one small model on one day, kept as a regression fixture: replaying them through the
current code must give the outcomes pinned in `tests/trial/test_recorded_trial.py`. That pins the behaviour of the
schema filter, the validator, the issue merging, the loop (corrections, the retry on the full schema, giving up,
declining) and the execution. It does not pin the wording of the prompts (a reply is not compared with the prompt it
answered); `tests/generation/test_prompts.py` and `tests/generation/test_prompt_rules.py` cover that. Record new replies (see
`evals/README.md`) when the prompts change on purpose and the old replies no longer fit.
