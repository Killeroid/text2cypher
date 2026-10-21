"""The entry point: ask a question of a graph."""

from dataclasses import replace

from falkordb.asyncio import FalkorDB
from redis.exceptions import RedisError

from text2cypher._config import Config
from text2cypher._execution import execute_cypher
from text2cypher._results import Result
from text2cypher._schema_source import SchemaSource
from text2cypher._validation import QueryValidator
from text2cypher._workflow import run_workflow
from text2cypher.llm import AsyncChatClient, AsyncLLMClient
from text2cypher.schema import SchemaFilter, filter_schema, schema_from_snapshot


class Text2Cypher:
    """Turns a question into a query that passes validation against a FalkorDB graph.

    Args:
        client: A ``falkordb.asyncio`` client. It is bound to one event loop, so use
            one ``Text2Cypher`` per loop.
        llm: The model that writes and corrects the queries. A client with ``chat`` is
            given each correction as the turns of a conversation; one with only
            ``complete``, as one message.
        config: The limits of the correction loop, whether to execute, and how long a
            graph's schema is kept.
        schema_filter: Keeps what a question needs of the graph's schema; by default
            ``filter_schema``, which matches the question's words to names. ``None``
            turns filtering off: every attempt uses the full schema.
    """

    def __init__(
        self,
        client: FalkorDB,
        llm: AsyncChatClient | AsyncLLMClient,
        *,
        config: Config | None = None,
        schema_filter: SchemaFilter | None = filter_schema,
    ) -> None:
        self._client = client
        self._llm = llm
        self._config = config or Config()
        self._schema_filter = schema_filter
        self._schemas = SchemaSource(client, self._config.schema_refresh_seconds)

    async def run(self, question: str, graph: str) -> Result:
        """Write a query for ``question`` over the graph named ``graph``.

        The schema is read from the graph on every call, or kept as
        ``config.schema_refresh_seconds`` says. With ``config.execute``, a
        valid query is also run, and the rows or the error are on the result: FalkorDB
        rejecting or failing the query, a timeout or a lost connection are all recorded
        there; any other error is a bug and propagates.

        Raises:
            GraphNotFoundError: If the graph does not exist (or has no labelled nodes or
                relationships), before any LLM call is made.
        """
        snapshot, source = await self._schemas.read(graph)
        # The validators answer from the snapshot the prompt was written from, so they cannot
        # disagree with it, and ask the database only to plan the query and confirm a problem
        validator = QueryValidator(
            self._client,
            snapshot=source,
            check_types=self._config.check_types,
            check_performance=self._config.check_performance,
        )
        result = await run_workflow(
            question,
            schema_from_snapshot(snapshot),
            llm=self._llm,
            validate=lambda query: validator.validate(query, graph),
            config=self._config,
            schema_filter=self._schema_filter,
        )
        if self._config.execute and result.status == "valid":
            return await self._execute(result, graph)
        return result

    async def _execute(self, result: Result, graph: str) -> Result:
        """``result`` with the rows its query returns, or why it did not run."""
        try:
            records = await execute_cypher(
                self._client.select_graph(graph),
                result.query,
                timeout_ms=self._config.execute_timeout_ms,
            )
        except RedisError as error:
            # The LLM calls are paid for: keep the valid query and say why it did not run
            return replace(result, execution_error=str(error))
        return replace(result, records=records)
