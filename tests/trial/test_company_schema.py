"""The large company schema of ``evals/datasets/company.py``: its gold queries fit it, and what the filter does on it."""

import pytest

from evals.dataset import Question
from evals.datasets.company import QUESTIONS, check_gold, measure, schema

pytestmark = pytest.mark.anyio


def test_the_schema_is_big_enough_to_say_something_about_the_filter():
    full = schema()

    assert (len(full.node_properties), len(full.relationship_properties)) == (36, 44)
    assert len(full.paths) == 46


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda question: question.id)
async def test_every_gold_query_fits_the_schema(question: Question):
    assert await check_gold(question) == []


async def test_a_gold_query_that_does_not_fit_is_reported():
    wrong = Question("x", "x", "text", "MATCH (e:Employee) RETURN e.shoe_size")

    problems = await check_gold(wrong)

    assert any("shoe_size" in problem for problem in problems)


def test_what_the_filter_keeps_of_the_large_schema_and_what_it_misses():
    # Recorded, not asserted to be good: a change to the filter shows up here. Of 80 names the
    # filter keeps 2 to 33 when a word of the question matches a name. "staff members know
    # Python" matches MEMBER_OF and so does not fall back to the whole schema, but misses
    # Skill and HAS_SKILL: the case the retry on the full schema is for.
    notes = {question.id: note for question, note in measure()}

    assert notes == {
        "l01": "25/80",
        "l02": "2/80",
        "l03": "28/80",
        "l04": "33/80",
        "l05": "15/80",
        "l06": "7/80",
        "l07": "19/80",
        "l08": "6/80",
        "l09": "17/80",
        "l10": "3/80 MISSING HAS_SKILL,Skill",
        "l11": "whole",
        "l12": "13/80",
        "l13": "whole",
        "l14": "3/80",
        "l15": "5/80",
        "l16": "28/80",
    }
