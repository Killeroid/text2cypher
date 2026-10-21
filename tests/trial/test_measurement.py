"""The datasets and the replay of the measurement of the LLM schema filter and of skills."""

import json
from pathlib import Path

import pytest
from falkordb.asyncio import FalkorDB

from evals.dataset import Question
from evals.datasets.company import check_gold
from evals.datasets.company_synonyms import COMPANY_SYNONYMS
from evals.datasets.skills import SKILLS
from evals.metrics import summarize
from evals.replay import (
    Condition,
    Finished,
    NeedReply,
    Pending,
    ReplayLLM,
    load_graph,
    replay,
)
from evals.trial import write_gold
from text2cypher import Tool

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "question", COMPANY_SYNONYMS.questions, ids=lambda question: question.id
)
async def test_every_gold_query_of_the_reworded_company_questions_fits_its_schema(
    question: Question,
):
    assert await check_gold(question) == []


async def test_the_gold_queries_of_the_skills_questions_answer_them_on_the_graph(
    falkordb_client: FalkorDB, tmp_path: Path
):
    await load_graph(falkordb_client, SKILLS)
    write_gold(tmp_path, SKILLS)

    items = await replay(falkordb_client, tmp_path, SKILLS.questions, SKILLS)

    assert all(isinstance(item, Finished) for item in items)
    assert {item.verdict for item in items if isinstance(item, Finished)} == {"OK"}


async def test_a_reply_of_call_lines_is_the_model_calling_a_tool():
    llm = ReplayLLM(['CALL read_skill {"id": "path-finding"}'])
    tool = Tool("read_skill", "Read one skill.", {"type": "object"})

    with pytest.raises(NeedReply):
        await ReplayLLM([]).chat("system", [], [tool])
    completion = await llm.chat("system", [], [tool])

    assert [(call.name, call.arguments) for call in completion.tool_calls] == [
        ("read_skill", {"id": "path-finding"})
    ]
    assert completion.text == ""


async def test_the_llm_filter_replies_are_the_first_replies_and_its_kept_names_are_noted(
    falkordb_client: FalkorDB, tmp_path: Path
):
    await load_graph(falkordb_client, COMPANY_SYNONYMS)
    question = COMPANY_SYNONYMS.questions[1]  # clients, a balance: Customer, Account
    kept = {
        "labels": ["Customer", "Account"],
        "relationship_types": ["HAS_ACCOUNT"],
    }
    (tmp_path / f"{question.id}_1.txt").write_text(json.dumps(kept))
    condition = Condition(filter="llm")

    first = await replay(
        falkordb_client, tmp_path, [question], COMPANY_SYNONYMS, condition
    )
    assert question.gold is not None
    (tmp_path / f"{question.id}_2.txt").write_text(question.gold)
    second = await replay(
        falkordb_client, tmp_path, [question], COMPANY_SYNONYMS, condition
    )

    assert isinstance(first[0], Pending)
    assert isinstance(second[0], Finished)
    assert second[0].replies == 2
    assert second[0].filter_note == "17/80"  # as measured; the filter adds to the reply


MEASURED = Path(__file__).parents[2] / "evals" / "recorded" / "measurement"

# cell: (dataset, condition, questions answered correctly, right on the first query)
CELLS = {
    "haiku_lexical": (COMPANY_SYNONYMS, Condition(), 20, 12),
    "haiku_llm_filter": (COMPANY_SYNONYMS, Condition(filter="llm"), 20, 20),
    "sonnet_lexical": (COMPANY_SYNONYMS, Condition(), 20, 11),
    "sonnet_llm_filter": (COMPANY_SYNONYMS, Condition(filter="llm"), 20, 19),
    "haiku_skills_off": (SKILLS, Condition(), 10, 8),
    "haiku_skills_on": (SKILLS, Condition(skills=True), 12, 12),
    "sonnet_skills_off": (SKILLS, Condition(), 15, 15),
    "sonnet_skills_on": (SKILLS, Condition(skills=True), 15, 15),
}


@pytest.mark.parametrize("cell", CELLS)
async def test_the_recorded_replies_of_each_measured_cell_give_the_recorded_numbers(
    cell: str, falkordb_client: FalkorDB
):
    dataset, condition, correct, first_try = CELLS[cell]
    await load_graph(falkordb_client, dataset)

    items = await replay(
        falkordb_client, MEASURED / cell, dataset.questions, dataset, condition
    )

    assert all(isinstance(item, Finished) for item in items)
    metrics = summarize([item for item in items if isinstance(item, Finished)])
    assert (metrics.correct, metrics.first_try) == (correct, first_try)
