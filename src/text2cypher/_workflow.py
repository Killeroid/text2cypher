"""The loop: generate a query, validate it, correct it, and escalate the schema."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from text2cypher._config import Config
from text2cypher._generation import Generation, correct_cypher, generate_cypher
from text2cypher._results import Attempt, Result, SchemaScope
from text2cypher._validation import Issue, ValidationResult
from text2cypher.llm import AsyncLLMClient
from text2cypher.schema import GraphSchema, SchemaFilter


@dataclass(frozen=True, slots=True)
class _Failed:
    """A query that did not pass validation, and why, to be corrected."""

    query: str
    issues: tuple[Issue, ...]


async def run_workflow(
    question: str,
    schema: GraphSchema,
    *,
    llm: AsyncLLMClient,
    validate: Callable[[str], Awaitable[ValidationResult]],
    config: Config,
    schema_filter: SchemaFilter | None = None,
) -> Result:
    """Answer ``question`` with a query that passes ``validate``, if one can be found.

    The first query and the first ``config.attempts_w_filtered`` corrections use the
    filtered schema, the later corrections the full one, until ``config.total_attempts``
    corrections have been made. Without a ``schema_filter``, or when it keeps the whole
    schema, there is no filtered schema and every attempt uses the full one.

    An empty query is the LLM declining. With the filtered schema that may be the
    filter's doing, so the same request is made again with the full schema, once and
    without costing a correction. Declining on the full schema ends the loop.
    """
    filtered = None if schema_filter is None else schema_filter(schema, question)
    if filtered == schema:
        filtered = None
    attempts: list[Attempt] = []
    # What the next request corrects; None for the first query
    failed: _Failed | None = None
    corrections = 0
    # The LLM declined on the filtered schema, which is no longer used
    escalated = False

    while True:
        scope, used = _choose_schema(
            schema, filtered, config, corrections=corrections, escalated=escalated
        )
        generation = await _write(
            llm, question, used, failed, parameterize=config.use_parameters
        )
        if not generation.query:
            attempts.append(Attempt("", None, scope, generation.usage))
            if scope == "full":
                return Result(question, "unanswerable", tuple(attempts))
            escalated = True
            continue

        validation = await validate(generation.query)
        attempts.append(Attempt(generation.query, validation, scope, generation.usage))
        if validation.valid:
            return Result(question, "valid", tuple(attempts))
        if corrections == config.total_attempts:
            return Result(question, "failed", tuple(attempts))
        corrections += 1
        failed = _Failed(generation.query, validation.issues)


def _choose_schema(
    schema: GraphSchema,
    filtered: GraphSchema | None,
    config: Config,
    *,
    corrections: int,
    escalated: bool,
) -> tuple[SchemaScope, GraphSchema]:
    """The filtered schema while there is one, the budget for it lasts and the LLM has
    not declined with it; the full schema otherwise."""
    if (
        filtered is not None
        and not escalated
        and corrections <= config.attempts_w_filtered
    ):
        return "filtered", filtered
    return "full", schema


async def _write(
    llm: AsyncLLMClient,
    question: str,
    schema: GraphSchema,
    failed: _Failed | None,
    *,
    parameterize: bool,
) -> Generation:
    """The first query, or a correction of the one that failed."""
    if failed is None:
        return await generate_cypher(llm, schema, question, parameterize=parameterize)
    return await correct_cypher(
        llm, schema, question, failed.query, failed.issues, parameterize=parameterize
    )
