"""Replaying the recorded trial replies gives the recorded outcomes.

The replies in ``evals/recorded/`` answer the prompts of one day. Replaying them through the
current code exercises the prompts, the schema filter, the validator, the corrector, the
retry on the full schema and the execution together, so a refactoring that changes any of
them shows up here. See ``evals/recorded/README.md``.
"""

from pathlib import Path

import pytest
from cypher_guardrail import issues as cg
from falkordb.asyncio import FalkorDB

from evals.dataset import Question, Seeded
from evals.datasets.movies import QUESTIONS, SEEDED
from evals.metrics import Metrics, summarize
from evals.replay import Finished, load_graph, replay
from evals.trial import scopes_of

pytestmark = pytest.mark.anyio

RECORDED = Path(__file__).parents[2] / "evals" / "recorded"

# (question id, status, schema of each attempt: f filtered, F full, verdict)
BASE = [
    ("q01", "valid", "f", "OK"),
    ("q02", "valid", "f", "OK"),
    ("q03", "valid", "f", "OK"),
    ("q04", "valid", "f", "OK"),
    ("q05", "valid", "F", "OK"),
    ("q06", "valid", "f", "OK"),
    ("q07", "valid", "f", "OK"),
    ("q08", "valid", "f", "OK (extra columns)"),
    ("q09", "valid", "f", "OK"),
    ("q10", "valid", "f", "OK"),
    ("q11", "unanswerable", "F", "OK"),
    ("q12", "unanswerable", "fF", "OK"),
    ("q13", "unanswerable", "fF", "OK"),
    ("q14", "unanswerable", "fF", "OK"),
    ("q15", "unanswerable", "F", "OK"),
    ("h01", "valid", "F", "OK"),
    ("h02", "valid", "f", "OK"),
    ("h03", "valid", "f", "OK"),
    ("h04", "valid", "f", "OK"),
    ("h05", "valid", "F", "OK"),
    ("h06", "valid", "f", "OK"),
    ("h07", "valid", "f", "OK"),
    ("h08", "valid", "F", "OK"),
    ("h09", "valid", "f", "OK"),
]
SEEDED_OUTCOMES = [
    ("s01", "valid", "ff", "OK"),
    ("s02", "valid", "ff", "OK"),
    ("s03", "valid", "ff", "OK"),
    ("s04", "valid", "ff", "OK"),
    ("s05", "valid", "FF", "OK"),
    ("s06", "unanswerable", "ffF", "OK"),
    ("s07", "valid", "ff", "OK"),
    ("s08", "valid", "FF", "OK"),
    ("s09", "valid", "ff", "OK"),
]


async def finished(
    client: FalkorDB,
    directory: str,
    questions: tuple[Question, ...] | tuple[Seeded, ...],
) -> list[Finished]:
    await load_graph(client)
    items = await replay(client, RECORDED / directory, questions)
    assert all(isinstance(item, Finished) for item in items)
    return [item for item in items if isinstance(item, Finished)]


def outcomes(items: list[Finished]) -> list[tuple[str, str, str, str]]:
    return [
        (item.question.id, item.result.status, scopes_of(item.result), item.verdict)
        for item in items
    ]


async def test_the_recorded_base_replies_give_the_recorded_outcomes(
    falkordb_client: FalkorDB,
):
    items = await finished(falkordb_client, "base", QUESTIONS)

    assert outcomes(items) == BASE
    assert summarize(items) == Metrics(
        answerable=19,
        correct=19,
        first_try=19,
        needed_correction=0,
        healed=0,
        corrections_to_heal=0,
        declinable=5,
        declined=5,
        issues_by_code={},
    )


async def test_the_recorded_seeded_replies_give_the_recorded_outcomes(
    falkordb_client: FalkorDB,
):
    items = await finished(falkordb_client, "seeded", SEEDED)

    assert outcomes(items) == SEEDED_OUTCOMES
    metrics = summarize(items)
    # Every flawed first query was healed by one correction; each issue code was reported once,
    # except two property and two relationship-type issues
    assert (metrics.needed_correction, metrics.healed, metrics.corrections_to_heal) == (
        8,
        8,
        8,
    )
    assert sum(metrics.issues_by_code.values()) == 10
    assert metrics.issues_by_code[cg.UNKNOWN_REL_TYPE] == 2
