"""The entry point: ask a question of a graph."""

from collections.abc import Awaitable, Callable
from dataclasses import replace

from falkordb.asyncio import FalkorDB

from text2cypher._config import Config
from text2cypher._execution import Execution, run_query
from text2cypher._generation import AsyncSchemaFilter, LLMSchemaFilter
from text2cypher._results import Result
from text2cypher._schema_source import SchemaSource
from text2cypher._validation import QueryValidator
from text2cypher._workflow import run_workflow
from text2cypher.llm import AsyncChatClient, AsyncLLMClient
from text2cypher.schema import SchemaFilter, filter_schema


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
            ``filter_schema``, which matches the question's words to names. An
            ``LLMSchemaFilter`` asks a model instead (it is given ``llm`` unless it has
            its own), and runs alone. ``None`` turns filtering off: every attempt uses the
            full schema. It may be a function or an async one; see ``AsyncSchemaFilter``.
    """

    def __init__(
        self,
        client: FalkorDB,
        llm: AsyncChatClient | AsyncLLMClient,
        *,
        config: Config | None = None,
        schema_filter: SchemaFilter | AsyncSchemaFilter | None = filter_schema,
    ) -> None:
        self._client = client
        self._llm = llm
        self._config = config or Config()
        # A filter that has no model of its own is given the main one
        if isinstance(schema_filter, LLMSchemaFilter) and schema_filter.llm is None:
            schema_filter = replace(schema_filter, llm=llm)
        self._schema_filter = schema_filter
        self._schemas = SchemaSource(
            client, self._config.schema_refresh_seconds, self._config.example_values
        )

    def clear_schema(self, graph: str | None = None) -> None:
        """Forget what is kept of the schema of ``graph``, or of every graph.

        The next ``run`` reads the schema again, and the example values too, which are kept
        while the structure of the graph is the same and so do not notice a changed value.
        Use it after changing a graph when ``config.schema_refresh_seconds`` keeps the schema,
        or to read new example values. It queries nothing and has no effect on a run that is
        in progress.
        """
        self._schemas.forget(graph)

    async def run(self, question: str, graph: str) -> Result:
        """Write a query for ``question`` over the graph named ``graph``.

        The schema is read from the graph on every call, or kept as
        ``config.schema_refresh_seconds`` says. With ``config.execute``, a query that
        passed validation is also run, read-only. If FalkorDB rejects or fails it, that
        error is what the next correction is told, and the result is ``failed`` when the
        corrections run out. The rows are on the result; a timeout or a lost connection
        ends the loop with the valid query and the error on the result. Any other error
        is a bug and propagates.

        Raises:
            GraphNotFoundError: If the graph does not exist (or has no labelled nodes or
                relationships), before any LLM call is made.
        """
        schema, source = await self._schemas.read(graph)
        # The validators answer from the snapshot the prompt was written from, so they cannot
        # disagree with it, and ask the database only to plan the query and confirm a problem
        validator = QueryValidator(
            self._client,
            snapshot=source,
            check_types=self._config.check_types,
            check_performance=self._config.check_performance,
        )
        return await run_workflow(
            question,
            schema,
            llm=self._llm,
            validate=lambda query: validator.validate(query, graph),
            config=self._config,
            schema_filter=self._schema_filter,
            execute=self._executor(graph),
        )

    def _executor(self, graph: str) -> Callable[[str], Awaitable[Execution]] | None:
        """What runs a query read-only on ``graph``, or ``None`` if execution is off."""
        if not self._config.execute:
            return None
        handle = self._client.select_graph(graph)
        return lambda query: run_query(
            handle, query, timeout_ms=self._config.execute_timeout_ms
        )
