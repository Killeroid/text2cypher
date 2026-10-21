"""Where the snapshot of a graph comes from: a new capture, or one that is kept."""

import time
from dataclasses import replace
from datetime import timedelta

from cypher_guardrail import AsyncLiveSnapshot, Snapshot, SnapshotSource
from falkordb.asyncio import FalkorDB

from text2cypher._errors import GraphNotFoundError
from text2cypher._examples import Examples, sample_examples
from text2cypher.schema import GraphSchema, schema_from_snapshot

# A live snapshot refreshes itself in the background after this long; ours is refreshed by
# ``SchemaSource.read`` when it is due, so it must never be due on its own.
_NEVER = timedelta(days=36500)


class SchemaSource:
    """Reads the schema of a graph that the prompt is made from, and the snapshot the
    validators answer from.

    Args:
        client: A ``falkordb.asyncio`` client.
        refresh_seconds: ``None`` captures a new snapshot on every read. A number keeps one
            per graph and, when that many seconds have passed since it was checked, checks
            it again: a few cheap queries, and a new capture only if the graph changed.
        example_values: How many example values of each property the schema has; ``0``
            reads none. They are read once for a structure of the graph: while a new snapshot
            has the same structure as the one they were read for, they are kept.
    """

    def __init__(
        self, client: FalkorDB, refresh_seconds: float | None, example_values: int = 0
    ) -> None:
        self._client = client
        self._refresh_seconds = refresh_seconds
        self._example_values = example_values
        # graph name -> its live snapshot and when it was last checked (monotonic seconds)
        self._live: dict[str, tuple[AsyncLiveSnapshot, float]] = {}
        # graph name -> the structure the examples were read for, and the examples
        self._examples: dict[str, tuple[Snapshot, tuple[Examples, Examples]]] = {}

    def forget(self, graph: str | None = None) -> None:
        """Drop what is kept for ``graph``, or for every graph: the next read starts again.

        That is the snapshot and the example values; nothing is queried until then.
        """
        for kept in (self._live, self._examples):
            if graph is None:
                kept.clear()
            else:
                kept.pop(graph, None)

    async def read(self, graph: str) -> tuple[GraphSchema, SnapshotSource]:
        """The graph's schema, and the source the validators should use.

        Raises:
            GraphNotFoundError: If the graph does not exist, or has no labelled nodes or
                relationships. FalkorDB has no graph without data, so this is a missing
                graph or a typo. Nothing is kept for it, so one created later is found.
        """
        snapshot, source = await self._snapshot(graph)
        if not snapshot.labels and not snapshot.relationship_types:
            self._live.pop(graph, None)
            self._examples.pop(graph, None)
            raise GraphNotFoundError(
                f"The graph {graph!r} does not exist, or has no labelled nodes or relationships"
            )
        return await self._schema(graph, snapshot), source

    async def _schema(self, graph: str, snapshot: Snapshot) -> GraphSchema:
        schema = schema_from_snapshot(snapshot)
        if self._example_values == 0:
            return schema
        # Two captures of one structure differ in when they were made, and in nothing else
        structure = replace(snapshot, captured_at=None)
        kept = self._examples.get(graph)
        if kept is None or kept[0] != structure:
            examples = await sample_examples(
                self._client, graph, schema, self._example_values
            )
            kept = self._examples[graph] = (structure, examples)
        nodes, relationships = kept[1]
        return replace(schema, node_examples=nodes, relationship_examples=relationships)

    async def _snapshot(self, graph: str) -> tuple[Snapshot, SnapshotSource]:
        if self._refresh_seconds is None:
            snapshot = await Snapshot.capture_async(self._client, graph, types=True)
            return snapshot, snapshot
        live, checked = self._live.get(graph) or (
            AsyncLiveSnapshot(self._client, graph, refresh_every=_NEVER, types=True),
            -float("inf"),
        )
        if time.monotonic() - checked >= self._refresh_seconds:
            await live.refresh()
            checked = time.monotonic()
        self._live[graph] = (live, checked)
        return live.current(), live
