"""Run a query that passed validation, read-only."""

from falkordb.asyncio.graph import AsyncGraph

from text2cypher._results import Records


async def execute_cypher(
    graph: AsyncGraph, query: str, *, timeout_ms: int | None = None
) -> Records:
    """The rows ``query`` returns from ``graph``.

    It runs as a read-only query, which FalkorDB rejects if it could write. Values
    are the client's: nodes, relationships and paths stay ``falkordb`` objects.

    Raises:
        redis.exceptions.RedisError: If FalkorDB rejects or fails the query (a
            ``ResponseError``, also when it runs longer than ``timeout_ms``) or cannot be
            reached.
    """
    result = await graph.ro_query(query, timeout=timeout_ms)
    return Records(
        columns=tuple(name for _, name in result.header),
        rows=tuple(tuple(row) for row in result.result_set),
    )
