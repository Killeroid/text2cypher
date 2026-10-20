"""Judge what the pipeline returned: the rows against the gold query's, and the filter's coverage."""

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from falkordb.asyncio.graph import AsyncGraph

from evals.dataset import Question, Seeded
from text2cypher import Result
from text2cypher.schema import GraphSchema, filter_schema

# One entry per row: its values as strings, sorted. The list is sorted too, so rows are
# compared by value and a duplicate row is a row of its own.
type Rows = list[tuple[str, ...]]


def normalize_rows(rows: Sequence[Sequence[Any]]) -> Rows:
    return sorted(tuple(sorted(str(value) for value in row)) for row in rows)


def _contains(row: tuple[str, ...], wanted: tuple[str, ...]) -> bool:
    return not Counter(wanted) - Counter(row)


def _covers(got: Rows, want: Rows) -> bool:
    """Whether every wanted row can be given a returned row of its own that contains its values.

    This is a matching between two sets of rows, found by augmenting paths: taking the first
    returned row that fits can leave a later wanted row with none, although another
    assignment works.
    """
    if len(got) != len(want):
        return False
    owner: dict[
        int, int
    ] = {}  # index of a returned row -> index of the wanted row it covers

    def assign(wanted: int, tried: set[int]) -> bool:
        for index, row in enumerate(got):
            if index not in tried and _contains(row, want[wanted]):
                tried.add(index)
                if index not in owner or assign(owner[index], tried):
                    owner[index] = wanted
                    return True
        return False

    return all(assign(wanted, set()) for wanted in range(len(want)))


def judge_rows(got: Rows, want: Rows) -> str:
    """Whether ``got`` answers the question that ``want`` is the answer to.

    Rows are compared by their values, not their order, and a duplicate row counts. A row
    with more values than the gold row (a rating next to the title that was asked for) is
    accepted, as long as every gold row has a returned row of its own and no row is left.
    """
    if got == want:
        return "OK"
    if _covers(got, want):
        return "OK (extra columns)"
    return f"MISMATCH got={str(got)[:90]} want={str(want)[:90]}"


async def judge(
    graph: AsyncGraph,
    question: Question | Seeded,
    result: Result,
    *,
    compare_rows: bool = True,
) -> str:
    if question.gold is None:
        if result.status == "unanswerable":
            return "OK"
        return f"WRONG (expected unanswerable, got {result.status})"
    if result.status != "valid":
        return f"no valid query ({result.status})"
    if result.execution_error:
        return f"EXEC ERROR {result.execution_error[:80]}"
    if not compare_rows:
        return "OK (valid, rows not compared)"
    if result.records is None:
        return "no rows"
    want = normalize_rows((await graph.ro_query(question.gold)).result_set)
    return judge_rows(normalize_rows(result.records.rows), want)


def _names(schema: GraphSchema) -> set[str]:
    return set(schema.node_properties) | set(schema.relationship_properties)


def filter_note(schema: GraphSchema, question: Question | Seeded) -> str:
    """How much of the schema the filter kept, and whether the gold query needed more."""
    kept = filter_schema(schema, question.text)
    note = "whole" if kept == schema else f"{len(_names(kept))}/{len(_names(schema))}"
    # Any word of the gold text that is a name in the schema: this also finds a
    # relationship type that is only written in a string, as in relTypes: ['KNOWS']
    needed = set(re.findall(r"[A-Za-z_]\w*", question.gold or "")) & _names(schema)
    if missing := needed - _names(kept):
        note += " MISSING " + ",".join(sorted(missing))
    return note
