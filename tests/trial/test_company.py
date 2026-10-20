"""The company graph as a real graph: the shape of the schema, with no meaningful rows."""

from pathlib import Path

import pytest
from cypher_guardrail import Snapshot
from falkordb.asyncio import FalkorDB

from evals.datasets.company import COMPANY, QUESTIONS, SEEDED, schema
from evals.replay import Finished, Pending, load_graph, replay
from evals.trial import write_gold, write_seed
from text2cypher.schema import schema_from_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
async def company(falkordb_client: FalkorDB) -> FalkorDB:
    await load_graph(falkordb_client, COMPANY)
    return falkordb_client


async def test_the_loaded_graph_has_the_schema_that_was_written_by_hand(
    company: FalkorDB,
):
    snapshot = await Snapshot.capture_async(company, COMPANY.graph, types=True)

    assert schema_from_snapshot(snapshot) == schema()


async def test_every_gold_query_passes_validation_on_the_real_graph(
    company: FalkorDB, tmp_path: Path
):
    write_gold(tmp_path, COMPANY)

    items = await replay(company, tmp_path, QUESTIONS, COMPANY)

    assert all(isinstance(item, Finished) for item in items)
    assert {item.verdict for item in items if isinstance(item, Finished)} == {
        "OK (valid, rows not compared)"
    }


async def test_every_flawed_first_query_waits_for_a_correction(
    company: FalkorDB, tmp_path: Path
):
    write_seed(tmp_path, COMPANY)

    items = await replay(company, tmp_path, SEEDED, COMPANY)

    assert all(isinstance(item, Pending) and item.number == 2 for item in items)
