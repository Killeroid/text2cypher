"""Run a query that passed validation, read-only."""

from dataclasses import dataclass

from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import RedisError, ResponseError

from text2cypher._results import Records


@dataclass(frozen=True, slots=True)
class Execution:
    """How running a query went: the rows it returned, or the error FalkorDB gave.

    ``correctable`` says whether another query might not fail: FalkorDB rejected or
    failed this one. A timeout, or a connection that is lost, says nothing about the query.
    """

    records: Records | None = None
    error: str | None = None
    correctable: bool = False


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


async def run_query(
    graph: AsyncGraph, query: str, *, timeout_ms: int | None = None
) -> Execution:
    """``execute_cypher``, with an error from FalkorDB or the connection as a value.

    Any other error is a bug and propagates.
    """
    try:
        records = await execute_cypher(graph, query, timeout_ms=timeout_ms)
    except RedisError as error:
        return Execution(error=str(error), correctable=_is_the_querys_fault(error))
    return Execution(records=records)


def _is_the_querys_fault(error: RedisError) -> bool:
    return isinstance(error, ResponseError) and "timed out" not in str(error).lower()
