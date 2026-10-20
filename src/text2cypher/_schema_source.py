"""Where the snapshot of a graph comes from: a new capture, or one that is kept."""

import time
from datetime import timedelta

from cypher_guardrail import AsyncLiveSnapshot, Snapshot, SnapshotSource
from falkordb.asyncio import FalkorDB

from text2cypher._errors import GraphNotFoundError

# A live snapshot refreshes itself in the background after this long; ours is refreshed by
# ``SchemaSource.read`` when it is due, so it must never be due on its own.
_NEVER = timedelta(days=36500)


class SchemaSource:
    """Reads the snapshot of a graph that the prompt and the validators are made from.

    Args:
        client: A ``falkordb.asyncio`` client.
        refresh_seconds: ``None`` captures a new snapshot on every read. A number keeps one
            per graph and, when that many seconds have passed since it was checked, checks
            it again: a few cheap queries, and a new capture only if the graph changed.
    """

    def __init__(self, client: FalkorDB, refresh_seconds: float | None) -> None:
        self._client = client
        self._refresh_seconds = refresh_seconds
        # graph name -> its live snapshot and when it was last checked (monotonic seconds)
        self._live: dict[str, tuple[AsyncLiveSnapshot, float]] = {}

    async def read(self, graph: str) -> tuple[Snapshot, SnapshotSource]:
        """The graph's snapshot, and the source the validators should use.

        Raises:
            GraphNotFoundError: If the graph does not exist, or has no labelled nodes or
                relationships. FalkorDB has no graph without data, so this is a missing
                graph or a typo. Nothing is kept for it, so one created later is found.
        """
        snapshot, source = await self._snapshot(graph)
        if not snapshot.labels and not snapshot.relationship_types:
            self._live.pop(graph, None)
            raise GraphNotFoundError(
                f"The graph {graph!r} does not exist, or has no labelled nodes or relationships"
            )
        return snapshot, source

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
