"""The numbers a trial reports: what was answered, what was healed, what went wrong."""

from typing import Literal

from cypher_guardrail import issues as cg
from support import VALID, invalid

from evals.dataset import Question
from evals.metrics import summarize
from evals.replay import Finished
from text2cypher import Attempt, Result

ANSWERABLE = Question("a", "x", "text", "MATCH (n) RETURN n")
DECLINE = Question("d", "x", "text", None)


def finished(
    question: Question,
    verdict: str,
    *attempts: Attempt,
    status: Literal["valid", "unanswerable", "failed"] = "valid",
) -> Finished:
    return Finished(question, Result("text", status, attempts), verdict, "whole")


def attempt(validation=VALID, query: str = "MATCH (n) RETURN n") -> Attempt:
    return Attempt(query, validation, "filtered", None)


def test_the_success_rate_counts_answerable_questions_whose_rows_are_right():
    metrics = summarize(
        [
            finished(ANSWERABLE, "OK", attempt()),
            finished(ANSWERABLE, "OK (extra columns)", attempt()),
            finished(ANSWERABLE, "MISMATCH got=[] want=[]", attempt()),
            finished(ANSWERABLE, "no valid query (failed)", attempt(), status="failed"),
        ]
    )

    assert (metrics.answerable, metrics.correct) == (4, 2)
    assert metrics.success_rate == 0.5


def test_a_decline_is_counted_apart_from_the_answerable_questions():
    metrics = summarize(
        [
            finished(ANSWERABLE, "OK", attempt()),
            finished(DECLINE, "OK", attempt(None, ""), status="unanswerable"),
            finished(DECLINE, "WRONG (expected unanswerable, got valid)", attempt()),
        ]
    )

    assert (metrics.answerable, metrics.declinable, metrics.declined) == (1, 2, 1)


def test_the_self_healing_rate_is_the_share_of_bad_first_queries_that_ended_right():
    metrics = summarize(
        [
            finished(ANSWERABLE, "OK", attempt()),  # right first time: not counted
            finished(ANSWERABLE, "OK", attempt(invalid("x")), attempt()),  # healed
            finished(
                ANSWERABLE,
                "no valid query (failed)",
                attempt(invalid("x")),
                attempt(invalid("y")),
                status="failed",
            ),  # not healed
        ]
    )

    assert (metrics.needed_correction, metrics.healed) == (2, 1)
    assert metrics.corrections_to_heal == 1
    assert metrics.self_healing_rate == 0.5
    assert metrics.first_try == 1


def test_there_is_no_rate_without_anything_to_divide_by():
    metrics = summarize([finished(DECLINE, "OK", attempt(None, ""))])

    assert metrics.success_rate is None
    assert metrics.self_healing_rate is None


def test_issues_are_counted_by_code_over_every_attempt():
    metrics = summarize(
        [
            finished(
                ANSWERABLE,
                "OK",
                attempt(invalid("x", cg.UNKNOWN_LABEL)),
                attempt(invalid("y", cg.UNKNOWN_REL_TYPE)),
                attempt(),
            ),
            finished(ANSWERABLE, "OK", attempt(invalid("z", cg.UNKNOWN_LABEL))),
        ]
    )

    assert metrics.issues_by_code == {cg.UNKNOWN_LABEL: 2, cg.UNKNOWN_REL_TYPE: 1}
