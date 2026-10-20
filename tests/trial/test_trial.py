"""The trial harness in ``evals/`` keeps working: it needs no model to check itself."""

from collections.abc import Callable
from pathlib import Path

import pytest
from falkordb.asyncio import FalkorDB

from evals.dataset import Question
from evals.datasets.company import COMPANY
from evals.datasets.movies import QUESTIONS, SEEDED
from evals.judge import filter_note, judge_rows, normalize_rows
from evals.replay import (
    Finished,
    Pending,
    load_graph,
    replay,
    replies_for,
)
from evals.trial import format_report, main, write_gold, write_seed
from text2cypher.schema import GraphSchema
from text2cypher.schema import Path as SchemaPath

pytestmark = pytest.mark.anyio


@pytest.fixture
async def movies(falkordb_client: FalkorDB) -> FalkorDB:
    await load_graph(falkordb_client)
    return falkordb_client


async def test_the_gold_queries_answer_every_question_as_expected(
    movies: FalkorDB, tmp_path: Path
):
    write_gold(tmp_path)

    items = await replay(movies, tmp_path, QUESTIONS)

    assert all(isinstance(item, Finished) for item in items)
    assert [item.verdict for item in items if isinstance(item, Finished)] == [
        "OK"
    ] * len(QUESTIONS)


async def test_the_report_ends_with_the_metrics_only_when_every_question_finished(
    movies: FalkorDB, tmp_path: Path
):
    write_gold(tmp_path)
    done = await replay(movies, tmp_path, QUESTIONS)
    (tmp_path / "q01_1.txt").unlink()
    waiting = await replay(movies, tmp_path, QUESTIONS)

    assert "answered correctly: 19/19 (100%)" in format_report(done)
    assert "answered correctly" not in format_report(waiting)
    assert "1 question(s) waiting for a reply" in format_report(waiting)


async def test_a_flawed_first_query_waits_for_a_correction_that_names_the_problem(
    movies: FalkorDB, tmp_path: Path
):
    write_seed(tmp_path)

    items = await replay(movies, tmp_path, SEEDED)

    assert all(isinstance(item, Pending) and item.number == 2 for item in items)
    by_id = {item.id: item for item in items if isinstance(item, Pending)}
    assert "Actor" in by_id["s02"].user
    assert "stars" in by_id["s08"].user


async def test_a_question_without_a_reply_waits_for_the_first_one(
    movies: FalkorDB, tmp_path: Path
):
    [item, *_] = await replay(movies, tmp_path, QUESTIONS)

    assert isinstance(item, Pending)
    assert (item.id, item.number) == ("q01", 1)
    assert "What are the titles of all movies released after 1998?" in item.user


@pytest.mark.parametrize(
    ("got", "want", "verdict"),
    [
        ([["a", "b"]], [["a", "b"]], "OK"),
        ([["b", "a"]], [["a", "b"]], "OK"),
        ([[1, "Action"], [2, "Crime"]], [["Action", 1], ["Crime", 2]], "OK"),
        ([], [], "OK"),
        ([["Matrix", 8.7]], [["Matrix"]], "OK (extra columns)"),
        ([[1, "Action"], [2, "Crime"]], [["Action"], ["Crime"]], "OK (extra columns)"),
        ([["a"], ["a"]], [["a"], ["a"]], "OK"),
        ([["a", "x"], ["a", "y"]], [["a"], ["a"]], "OK (extra columns)"),
        # the first wanted row must not take the row the second one needs
        ([["a", "b"], ["a", "c"]], [["a"], ["b"]], "OK (extra columns)"),
    ],
)
def test_rows_are_judged_by_their_values_not_their_order_or_extra_columns(
    got: list[list[object]], want: list[list[object]], verdict: str
):
    assert judge_rows(normalize_rows(got), normalize_rows(want)) == verdict


@pytest.mark.parametrize(
    ("got", "want"),
    [
        ([["Matrix"]], [["Matrix", 8.7]]),  # fewer columns
        ([["Heat"]], [["Matrix"]]),  # other values
        ([["Matrix"], ["Heat"]], [["Matrix"]]),  # more rows
        ([["a", "x"], ["a", "y"]], [["a"]]),  # two rows for one wanted row
        (
            [["a", "b"], ["z"]],
            [["a"], ["b"]],
        ),  # one row covers both, another matches none
        ([], [["Matrix"]]),
        ([["Matrix"]], []),
        ([["a"], ["a"]], [["a"]]),  # a duplicate row is another row
        ([["a"]], [["a"], ["a"]]),
    ],
)
def test_rows_that_do_not_answer_the_question_are_a_mismatch(
    got: list[list[object]], want: list[list[object]]
):
    assert judge_rows(normalize_rows(got), normalize_rows(want)).startswith("MISMATCH")


def test_the_filter_note_sees_a_relationship_type_that_is_only_named_in_a_string():
    schema = GraphSchema(
        {"Person": {}, "Movie": {}, "Studio": {}},
        {"DIRECTED": {}, "KNOWS": {}},
        (
            SchemaPath("Person", "DIRECTED", "Movie"),
            SchemaPath("Studio", "KNOWS", "Studio"),
        ),
    )
    gold = (
        "CALL algo.SPpaths({sourceNode: a, relTypes: ['KNOWS']}) YIELD path RETURN path"
    )

    note = filter_note(
        schema, Question("x", "category", "Tell me about a person", gold)
    )

    assert note.endswith("MISSING KNOWS")


@pytest.mark.parametrize("write", [write_gold, write_seed])
def test_a_directory_that_already_has_replies_is_not_overwritten(
    tmp_path: Path, write: Callable[[Path], None]
):
    write_gold(tmp_path / "run")

    with pytest.raises(FileExistsError, match="not empty"):
        write(tmp_path / "run")


def test_a_missing_reply_number_is_an_error_not_a_silent_stall(tmp_path: Path):
    (tmp_path / "q01_1.txt").write_text("first", encoding="utf-8")
    (tmp_path / "q01_3.txt").write_text("third", encoding="utf-8")

    with pytest.raises(ValueError, match=r"q01.*reply 2 is missing"):
        replies_for(tmp_path, "q01")


def test_replies_are_read_in_order_up_to_the_first_missing_number(tmp_path: Path):
    (tmp_path / "q01_1.txt").write_text("first", encoding="utf-8")
    (tmp_path / "q01_2.txt").write_text("second", encoding="utf-8")

    assert replies_for(tmp_path, "q01") == ["first", "second"]
    assert replies_for(tmp_path, "q02") == []


def test_the_command_line_writes_the_gold_and_the_seed_replies(tmp_path: Path):
    main(["gold", str(tmp_path / "gold")])
    main(["seed", str(tmp_path / "seed")])

    # a reply per question, and a second empty one for each that should be declined
    assert len(list((tmp_path / "gold").glob("*.txt"))) == len(QUESTIONS) + 5
    assert len(list((tmp_path / "seed").glob("*.txt"))) == len(SEEDED)


def test_the_command_line_writes_the_replies_of_the_other_dataset(tmp_path: Path):
    main(["gold", str(tmp_path), "--dataset", "company"])

    assert {path.stem for path in tmp_path.glob("l*_1.txt")} == {
        question.id + "_1" for question in COMPANY.questions
    }


@pytest.mark.db  # it starts its own embedded database
def test_the_command_line_replays_the_replies_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    main(["gold", str(tmp_path)])

    main(["round", str(tmp_path)])

    report = capsys.readouterr().out
    assert "all questions finished" in report
    assert "answered correctly: 19/19 (100%)" in report


@pytest.mark.db  # it starts its own embedded database
def test_the_command_line_writes_the_prompts_that_wait_for_a_reply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    main(["seed", str(tmp_path)])

    main(["round", str(tmp_path), "--seeded"])

    assert "waiting for reply 2" in capsys.readouterr().out
    assert len(list((tmp_path / "pending").glob("s*_2.txt"))) == len(SEEDED)
