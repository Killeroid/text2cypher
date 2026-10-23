"""The runner of the format measurement, with a stand-in model: no key, no network."""

from typing import Any

import pytest

from evals.datasets import DATASETS
from evals.format_trial import (
    Cell,
    OracleLLM,
    matrix,
    report,
    run_cells,
    run_filter_cell,
    run_generator_cell,
    unsaved,
)
from evals.replay import trial_client
from text2cypher.llm import Message

pytestmark = pytest.mark.anyio

MOVIES = DATASETS["movies"]
SYNONYMS = DATASETS["company_synonyms"]


class Broken:
    """A model whose every call fails."""

    async def chat(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("the endpoint is down")


def test_the_matrix_has_every_combination_once():
    cells = matrix(["generator", "filter"])

    assert len(cells) == len({cell.name for cell in cells})
    assert len([c for c in cells if c.experiment == "generator"]) == 2 * 2 * 3 * 2
    assert len([c for c in cells if c.experiment == "filter"]) == 1 * 2 * 3 * 3
    # The filter is measured on the company schema alone, off, with B and with C
    assert {c.mode for c in cells if c.experiment == "filter"} == {"off", "B", "C"}
    assert {c.dataset for c in cells if c.experiment == "filter"} == {
        "company_synonyms"
    }


def test_the_matrix_can_be_narrowed():
    cells = matrix(["filter"], models=["cheap"], layouts=["xml"])

    assert {(c.model, c.layout) for c in cells} == {("cheap", "xml")}
    assert len(cells) == 3


async def test_a_generator_cell_asks_in_its_layout_with_or_without_the_dictionary():
    seen = {}
    async with trial_client(MOVIES) as database:
        for layout, mode in (("plain", "off"), ("markdown", "on"), ("xml", "on")):
            llm = OracleLLM(MOVIES)
            cell = Cell("generator", "movies", "cheap", layout, mode)
            rows = await run_generator_cell(
                cell, llm, database, MOVIES.questions[:3], 2
            )
            assert all(row["ok"] for row in rows), rows
            seen[layout] = llm.calls[0]

    plain_system, plain_user = seen["plain"]
    assert "Data dictionary" not in plain_system
    assert plain_user.startswith("Question:")
    markdown_system, markdown_user = seen["markdown"]
    assert "# Rules" in markdown_system
    assert "### Labels" in markdown_system
    assert markdown_user.startswith("# Question\n")
    xml_system, xml_user = seen["xml"]
    assert "<data_dictionary>" in xml_system
    assert xml_user.startswith("<question>")


async def test_every_question_is_asked_once_with_no_correction():
    async with trial_client(MOVIES) as database:
        llm = OracleLLM(MOVIES)
        cell = Cell("generator", "movies", "cheap", "plain", "off")

        rows = await run_generator_cell(cell, llm, database, MOVIES.questions[:5], 3)

    assert len(llm.calls) == 5
    assert [row["id"] for row in rows] == [q.id for q in MOVIES.questions[:5]]


async def test_a_wrong_first_query_is_not_corrected_and_is_not_ok():
    class Wrong(OracleLLM):
        async def chat(self, system: str, messages: Any, *_: Any, **__: Any) -> Any:
            completion = await super().chat(system, messages)
            return type(completion)(
                "```cypher\nMATCH (p:Robot) RETURN p.name\n```", completion.usage
            )

    async with trial_client(MOVIES) as database:
        llm = Wrong(MOVIES)
        cell = Cell("generator", "movies", "cheap", "plain", "off")

        rows = await run_generator_cell(cell, llm, database, [MOVIES.questions[0]], 1)

    assert len(llm.calls) == 1  # no correction was asked for
    assert rows[0]["ok"] is False
    assert rows[0]["status"] == "failed"


async def test_a_call_that_fails_is_an_error_row_and_never_a_success():
    async with trial_client(MOVIES) as database:
        cell = Cell("generator", "movies", "cheap", "xml", "on")

        rows = await run_generator_cell(
            cell, Broken(), database, MOVIES.questions[:2], 2
        )

    assert [row["ok"] for row in rows] == [False, False]
    assert all("error" in row for row in rows)


async def test_a_filter_whose_call_fails_is_an_error_and_not_a_right_answer():
    # LLMSchemaFilter keeps the whole schema when its call fails, which would look right
    cell = Cell("filter", "company_synonyms", "cheap", "xml", "C")

    rows = await run_filter_cell(cell, Broken(), SYNONYMS.questions[:2], 2)

    assert [row["ok"] for row in rows] == [False, False]
    assert all("not asked" in row["error"] for row in rows)


async def test_a_filter_cell_is_judged_by_the_names_the_gold_query_needs():
    cell = Cell("filter", "company_synonyms", "cheap", "markdown", "B")
    llm = OracleLLM(SYNONYMS)

    rows = await run_filter_cell(cell, llm, SYNONYMS.questions[:5], 2)

    assert all(row["ok"] for row in rows), rows
    assert all(row["kept_names"] < 80 for row in rows)
    assert "Data dictionary" in llm.calls[0][1] or "data_dictionary" in llm.calls[0][1]
    assert "Employee" in llm.calls[0][1]


async def test_a_filter_that_leaves_out_a_needed_name_is_not_right():
    class Forgetful(OracleLLM):
        async def chat(self, system: str, messages: Any, *_: Any, **__: Any) -> Any:
            completion = await super().chat(system, messages)
            return type(completion)(
                '```json\n{"labels": ["City"], "relationship_types": []}\n```',
                completion.usage,
            )

    cell = Cell("filter", "company_synonyms", "cheap", "plain", "off")

    rows = await run_filter_cell(cell, Forgetful(SYNONYMS), SYNONYMS.questions[:1], 1)

    assert rows[0]["ok"] is False
    assert "MISSING" in rows[0]["verdict"]


async def test_a_saved_cell_is_not_run_again_unless_asked_and_the_report_reads_what_was_saved(
    tmp_path,
):
    cells = matrix(["filter"], models=["cheap"], layouts=["plain", "xml"])

    await run_cells(
        unsaved(cells, tmp_path, force=False),
        {},
        out=tmp_path,
        limit=2,
        concurrency=2,
        oracle=True,
    )

    assert unsaved(cells, tmp_path, force=False) == []
    assert unsaved(cells, tmp_path, force=True) == cells
    text = report(tmp_path, {"cheap": "a-model"})
    assert "filter / company_synonyms / cheap (a-model)" in text
    assert "2/2 ok" in text
    assert "ERRORS" not in text


async def test_the_report_says_how_many_calls_failed(tmp_path):
    cell = Cell("filter", "company_synonyms", "cheap", "plain", "off")
    rows = await run_filter_cell(cell, Broken(), SYNONYMS.questions[:2], 2)
    from evals.format_trial import _save

    _save(tmp_path, cell, rows)

    assert "0/2 ok, 2 ERRORS" in report(tmp_path)


async def test_a_saved_cell_with_a_failed_call_is_run_again(tmp_path):
    from evals.format_trial import _save

    cell = Cell("filter", "company_synonyms", "cheap", "plain", "off")
    failed = await run_filter_cell(cell, Broken(), SYNONYMS.questions[:2], 2)
    _save(tmp_path, cell, failed)

    assert unsaved([cell], tmp_path, force=False) == [cell]

    worked = await run_filter_cell(cell, OracleLLM(SYNONYMS), SYNONYMS.questions[:2], 2)
    _save(tmp_path, cell, worked)

    assert unsaved([cell], tmp_path, force=False) == []


async def test_the_oracle_finds_a_question_that_is_part_of_another_or_escaped():
    from evals.dataset import Dataset, Question

    short = Question("a", "x", "Who acted?", "MATCH (a) RETURN a")
    long = Question("b", "x", "Who acted? In what year?", "MATCH (b) RETURN b")
    angle = Question("c", "x", "Is 1 < 2?", "MATCH (c) RETURN c")
    oracle = OracleLLM(Dataset("x", "", (short, long, angle), ()))

    async def gold(user: str) -> str:
        reply = await oracle.chat("system", [Message("user", user)])
        return reply.text

    assert "MATCH (b)" in await gold("# Question\nWho acted? In what year?")
    assert "MATCH (a)" in await gold("# Question\nWho acted?")
    assert "MATCH (c)" in await gold("<question>Is 1 &lt; 2?</question>")
    with pytest.raises(ValueError, match="no question"):
        await gold("# Question\nSomething else")
