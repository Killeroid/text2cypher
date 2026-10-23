"""The entry point: ask a question of a graph."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from falkordb.asyncio import FalkorDB
from openai import AsyncOpenAI

from text2cypher._config import Config
from text2cypher._generation import AsyncSchemaFilter, LLMSchemaFilter
from text2cypher._graph import Execution, SchemaSource, run_query
from text2cypher._results import Result
from text2cypher._skills import Skill, catalog, check_inline, skills
from text2cypher._validation import QueryValidator
from text2cypher._workflow import run_workflow
from text2cypher.llm import AsyncChatClient, AsyncLLMClient, ModelOptions
from text2cypher.llm._adapt import as_client
from text2cypher.llm._base import supports_tools
from text2cypher.schema import DataDictionary, SchemaFilter, filter_schema


class Text2Cypher:
    """Turns a question into a query that passes validation against a FalkorDB graph.

    Args:
        client: A ``falkordb.asyncio`` client. It is bound to one event loop, so use
            one ``Text2Cypher`` per loop.
        llm: The model that writes and corrects the queries: an ``openai.AsyncOpenAI`` (with
            ``model``), an ``OpenAIClient``, or a client of your own. A client with ``chat``
            is given each correction as the turns of a conversation; one with only
            ``complete``, as one message.
        model: The model to ask, when ``llm`` is an ``openai.AsyncOpenAI``, which has none
            of its own. Any other kind of ``llm`` chooses its own model, and ``model`` with
            one is an error.
        model_options: How to ask ``model``, for an ``openai.AsyncOpenAI``: the
            ``temperature``, ``reasoning_effort``, ``timeout`` and other settings of
            ``ModelOptions``; a setting left out takes the default of ``OpenAIClient``. Any
            other kind of ``llm`` has its own settings, and ``model_options`` with one is an
            error.
        config: The limits of the correction loop, whether to execute, and how long a
            graph's schema is kept.
        schema_filter: Keeps what a question needs of the graph's schema; by default
            ``filter_schema``, which matches the question's words to names. An
            ``LLMSchemaFilter`` asks a model instead (it is given ``llm`` unless it has
            its own), and runs alone. ``None`` turns filtering off: every attempt uses the
            full schema. It may be a function or an async one; see ``AsyncSchemaFilter``.
        skills: The skills the LLM can read, in place of the ones that ship (to add to them,
            pass ``[*builtin_skills(), yours]``; see ``text2cypher.skills``). It needs
            ``Config(use_skills=True)``. A skill is text that steers the model and goes to the
            LLM provider: use only text you trust. A client that cannot be offered tools is
            given every skill in each prompt, so they may add up to 16,000 characters.
        data_dictionary: What the labels, relationship types and properties of the graph mean,
            written by you; the prompts show it next to the schema and ``LLMSchemaFilter``
            reads it. It is not read from the graph, so it is never kept with the schema, and
            a name the graph does not have is left out of the prompt (``check`` says which).
            It goes to the LLM provider like the schema does: write only what you trust. The
            same dictionary is used for every graph the agent is asked about.

    Raises:
        ValueError: If ``skills`` is given without ``use_skills``, is empty, has two skills
            with one id, or (for a client without tools) is too long to put in every prompt.
    """

    def __init__(
        self,
        client: FalkorDB,
        llm: AsyncChatClient | AsyncLLMClient | AsyncOpenAI,
        *,
        model: str | None = None,
        model_options: ModelOptions | None = None,
        config: Config | None = None,
        schema_filter: SchemaFilter | AsyncSchemaFilter | None = filter_schema,
        skills: Sequence[Skill] | None = None,
        data_dictionary: DataDictionary | None = None,
    ) -> None:
        self._client = client
        self._data_dictionary = data_dictionary
        self._llm = as_client(llm, model, model_options)
        self._config = config or Config()
        self._skills = self._skills_on_offer(skills)
        # A filter that has no client of its own is given the main one
        if isinstance(schema_filter, LLMSchemaFilter):
            schema_filter = schema_filter.bound(self._llm)
        self._schema_filter = schema_filter
        self._schemas = SchemaSource(
            client, self._config.schema_refresh_seconds, self._config.example_values
        )

    def _skills_on_offer(
        self, chosen: Sequence[Skill] | None
    ) -> dict[str, Skill] | None:
        """What the LLM may read, checked; ``None`` when skills are off."""
        if not self._config.use_skills:
            if chosen is not None:
                raise ValueError("skills needs Config(use_skills=True)")
            return None
        offered = dict(skills()) if chosen is None else catalog(chosen)
        if not supports_tools(self._llm):
            check_inline(offered)
        return offered

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
        if self._data_dictionary is not None:
            schema = replace(schema, data_dictionary=self._data_dictionary)
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
            skills=self._skills,
        )

    def _executor(self, graph: str) -> Callable[[str], Awaitable[Execution]] | None:
        """What runs a query read-only on ``graph``, or ``None`` if execution is off."""
        if not self._config.execute:
            return None
        handle = self._client.select_graph(graph)
        return lambda query: run_query(
            handle, query, timeout_ms=self._config.execute_timeout_ms
        )
