"""A few example values of each property, to show the LLM how values are written."""

import asyncio
from collections.abc import Mapping

from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.exceptions import RedisError

from text2cypher.schema import GraphSchema
from text2cypher.schema._model import quote_identifier

type Examples = dict[str, dict[str, tuple[str, ...]]]

# Only values that read as plain text in a prompt; lists, maps, points and vectors do not
_SCALARS = {"String", "Integer", "Float", "Boolean"}
# The first non-null values read per property, of which the distinct ones are the
# examples: a read that stops after a few rows, where DISTINCT would read every one
_ROWS = 20
_LONGEST = 40
# A property that is slow to sample has no examples: they are a help, not a need
_TIMEOUT_MS = 2_000
_MAX_CONCURRENCY = 8


async def sample_examples(
    client: FalkorDB, graph: str, schema: GraphSchema, count: int
) -> tuple[Examples, Examples]:
    """Up to ``count`` example values of each scalar property of ``schema``: of the
    labels, then of the relationship types. A query that fails gives no examples.

    Each is one read-only query that stops after a few rows, with a timeout.
    """
    handle = client.select_graph(graph)
    gate = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def sample(query: str) -> tuple[str, ...]:
        async with gate:
            return await _first_values(handle, query, count)

    jobs = {
        (kind, entity, prop): sample(_query(kind, entity, prop))
        for kind, entities in (
            ("node", schema.node_properties),
            ("relationship", schema.relationship_properties),
        )
        for entity, properties in entities.items()
        for prop, types in properties.items()
        if set(types) <= _SCALARS
    }
    found = dict(zip(jobs, await asyncio.gather(*jobs.values()), strict=True))
    return (
        _group(found, "node"),
        _group(found, "relationship"),
    )


def _query(kind: str, entity: str, prop: str) -> str:
    name, key = quote_identifier(entity), quote_identifier(prop)
    pattern = f"(e:{name})" if kind == "node" else f"()-[e:{name}]->()"
    return f"MATCH {pattern} WHERE e.{key} IS NOT NULL RETURN e.{key} LIMIT {_ROWS}"


async def _first_values(graph: AsyncGraph, query: str, count: int) -> tuple[str, ...]:
    try:
        result = await graph.ro_query(query, timeout=_TIMEOUT_MS)
    except RedisError:
        return ()
    distinct = dict.fromkeys(_literal(row[0]) for row in result.result_set)
    return tuple(distinct)[:count]


def _group(
    found: Mapping[tuple[str, str, str], tuple[str, ...]], kind: str
) -> Examples:
    grouped: Examples = {}
    for (found_kind, entity, prop), values in found.items():
        if found_kind == kind and values:
            grouped.setdefault(entity, {})[prop] = values
    return grouped


def _literal(value: object) -> str:
    """``value`` as it is written in a query; a long string is cut short."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if not isinstance(value, str):
        return str(value)
    text = value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    return f"'{text[:_LONGEST]}…'" if len(text) > _LONGEST else f"'{text}'"
