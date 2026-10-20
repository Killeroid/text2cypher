"""Validate a Cypher query against a FalkorDB graph with cypher-guardrail."""

import asyncio

from cypher_guardrail import (
    AsyncPropertiesValidator,
    AsyncSchemaValidator,
    AsyncSyntaxValidator,
    SnapshotSource,
)
from cypher_guardrail import GraphNotFoundError as GuardrailGraphNotFoundError
from cypher_guardrail.backend import get_async_backend
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph

from text2cypher._errors import GraphNotFoundError
from text2cypher._validation.issues import (
    Issue,
    Source,
    ValidationResult,
    drop_repeated_property_issues,
)


class QueryValidator:
    """Runs cypher-guardrail's syntax, schema and properties validators on a query.

    All three always run, even when the syntax is invalid: they can still report
    real problems elsewhere in the query. The query is valid when the syntax is ok
    and no validator reported an issue; scores are never used to decide that. A
    query that can write to the graph is never valid (cypher-guardrail's default).

    Args:
        client: A ``falkordb.asyncio`` client, or one of its graphs. The client is
            bound to one event loop, so use one validator per loop.
        max_concurrency: The most requests in flight at once.
        snapshot: A snapshot (or a live one) of the graph. The validators then answer
            from it and ask the database only to plan the query and to confirm a problem
            before reporting it, so a query that is fine costs one request. The answers are
            the same as without it, unless the graph changed since the snapshot.
        check_types: Also report a property compared with a value of another type, and
            ``sum``, ``avg`` and the like of a property that holds no numbers.
        check_performance: Also report a valid query that may be expensive: a cartesian
            product, a scan of every node, a path with no upper bound.
    """

    def __init__(
        self,
        client: FalkorDB | AsyncGraph,
        *,
        max_concurrency: int = 8,
        snapshot: SnapshotSource | None = None,
        check_types: bool = False,
        check_performance: bool = False,
    ) -> None:
        # One backend shared by the validators: one snapshot cache, one concurrency limit
        self._backend = get_async_backend(client, max_concurrency, snapshot=snapshot)
        self._syntax = AsyncSyntaxValidator(
            self._backend, check_performance=check_performance
        )
        self._schema = AsyncSchemaValidator(self._backend)
        self._properties = AsyncPropertiesValidator(
            self._backend, check_types=check_types
        )

    async def validate(self, query: str, graph: str | None = None) -> ValidationResult:
        """Validate ``query`` against ``graph`` (optional when given a graph).

        Raises:
            ValueError: If there is no graph to validate against.
            GraphNotFoundError: If the graph does not exist.
        """
        self._backend.check_database(graph)
        try:
            with self._backend.snapshot():
                try:
                    # Unlike gather, a failure cancels the other validators before the
                    # snapshot is released
                    async with asyncio.TaskGroup() as group:
                        syntax = group.create_task(self._syntax.validate(query, graph))
                        schema = group.create_task(self._schema.validate(query, graph))
                        properties = group.create_task(
                            self._properties.validate(query, graph)
                        )
                except ExceptionGroup as failures:
                    # Callers should see the validator's own error, not a group
                    raise failures.exceptions[0] from None
        except GuardrailGraphNotFoundError as error:
            # The graph can go away after its schema was read
            raise GraphNotFoundError(str(error)) from error
        syntax_ok, syntax_issues = syntax.result()
        schema_score, schema_issues = schema.result()
        properties_score, properties_issues = properties.result()

        issues = [
            *_wrap("syntax", syntax_issues),
            *_wrap("schema", schema_issues),
            *_wrap("properties", properties_issues),
        ]
        return ValidationResult(
            issues=tuple(drop_repeated_property_issues(issues)),
            syntax_ok=syntax_ok,
            schema_score=_meaningful(schema_score, syntax_ok, schema_issues),
            properties_score=_meaningful(
                properties_score, syntax_ok, properties_issues
            ),
        )


def _wrap(source: Source, raw_issues: list) -> list[Issue]:
    return [Issue(source=source, raw=raw) for raw in raw_issues]


def _meaningful(score: float | None, syntax_ok: bool, raw_issues: list) -> float | None:
    """Drop a score from a validator that found nothing in an invalid query."""
    return score if syntax_ok or raw_issues else None
