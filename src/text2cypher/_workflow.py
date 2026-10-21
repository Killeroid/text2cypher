"""The loop: generate a query, validate it, correct it, and escalate the schema."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from redis.exceptions import RedisError

from text2cypher._config import Config
from text2cypher._execution import Execution
from text2cypher._generation import (
    Generation,
    Round,
    fix_typos,
    generate_cypher,
    repair_header,
)
from text2cypher._results import (
    Attempt,
    DatabaseError,
    LLMError,
    Result,
    SchemaScope,
)
from text2cypher._validation import ValidationResult
from text2cypher.llm import AsyncChatClient, AsyncLLMClient, Usage
from text2cypher.schema import GraphSchema, SchemaFilter

# A wrong label stops the validators from checking its properties, so a query with both can
# need a second round of fixes; a third is for the rare case of another layer. The header of a
# query with parameters is repaired first, because a syntax error hides every other problem
_FIX_ROUNDS = 3


async def run_workflow(
    question: str,
    schema: GraphSchema,
    *,
    llm: AsyncChatClient | AsyncLLMClient,
    validate: Callable[[str], Awaitable[ValidationResult]],
    config: Config,
    schema_filter: SchemaFilter | None = None,
    execute: Callable[[str], Awaitable[Execution]] | None = None,
) -> Result:
    """Answer ``question`` with a query that passes ``validate``, if one can be found.

    The first query and the first ``config.attempts_w_filtered`` corrections use the
    filtered schema, the later corrections the full one, until ``config.total_attempts``
    corrections have been made. Without a ``schema_filter``, or when it keeps the whole
    schema, there is no filtered schema and every attempt uses the full one.

    An empty query is the LLM declining. With the filtered schema that may be the
    filter's doing, so the same request is made again with the full schema, once and
    without costing a correction. Declining on the full schema ends the loop.

    Each correction is a request that carries every query written so far with what was
    wrong with it, so the LLM does not repeat a mistake it was told about.

    An exception from the LLM ends the run with an ``LLMError``, and one from FalkorDB while
    validating with a ``DatabaseError``; both keep the attempts and the tokens so far.

    With ``execute``, a query that passed validation is run. If FalkorDB rejects or fails
    it, that is what the next correction is told, and it costs a correction like any
    other; when the corrections run out the result is ``failed``. A timeout or a lost
    connection ends the loop with the valid query and the error, as they say nothing
    about the query.
    """
    filtered = None if schema_filter is None else schema_filter(schema, question)
    if filtered == schema:
        filtered = None
    attempts: list[Attempt] = []
    # The queries written so far that failed, which the next request corrects
    rounds: list[Round] = []
    corrections = 0
    # The LLM declined on the filtered schema, which is no longer used
    escalated = False

    while True:
        scope, used = _choose_schema(
            schema, filtered, config, corrections=corrections, escalated=escalated
        )
        rounds = _with_schema_note(rounds, attempts, scope)
        try:
            generation = await generate_cypher(
                llm, used, question, rounds, parameterize=config.use_parameters
            )
        except Exception as error:
            raise LLMError.after(question, attempts, error) from error
        if not generation.query:
            attempts.append(Attempt("", None, scope, generation.usage))
            if scope == "full":
                return Result(question, "unanswerable", tuple(attempts))
            escalated = True
            continue

        try:
            checked, validation = await _check(
                generation,
                scope,
                validate,
                schema,
                header=config.use_parameters,
                typos=config.autofix_typos,
            )
        except RedisError as error:
            # The query was written and paid for: it is kept, with no validation
            unchecked = Attempt(generation.query, None, scope, generation.usage)
            raise DatabaseError.after(
                question, [*attempts, unchecked], error
            ) from error
        query = checked[-1].query
        ran = Execution()
        if validation.valid and execute is not None:
            ran = await execute(query)
        attempts.extend(
            (*checked[:-1], replace(checked[-1], execution_error=ran.error))
        )
        if validation.valid and not ran.correctable:
            return Result(question, "valid", tuple(attempts), ran.records, ran.error)
        if corrections == config.total_attempts:
            return Result(question, "failed", tuple(attempts), None, ran.error)
        corrections += 1
        rounds.append(Round(query, validation.issues, ran.error))


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


async def _check(
    generation: Generation,
    scope: SchemaScope,
    validate: Callable[[str], Awaitable[ValidationResult]],
    schema: GraphSchema,
    *,
    header: bool,
    typos: bool,
) -> tuple[list[Attempt], ValidationResult]:
    """The attempt for the query the LLM wrote, validated, and the validation that counts.

    A query that did not pass can be fixed without the LLM and validated again, up to a few
    times because fixing one thing can show another: with ``header``, its parameter header
    is written the way FalkorDB wants it (that changes nothing about what the query means,
    so it needs no option), and with ``typos`` its misspelled names are replaced (with the
    full schema). If it passes, it follows as an attempt of its own and its validation is
    the one that counts; if it never does, it is dropped and the query the LLM wrote stands.
    """
    validation = await validate(generation.query)
    attempt = Attempt(generation.query, validation, scope, generation.usage)
    if validation.valid or not (header or typos):
        return [attempt], validation
    query, now = generation.query, validation
    for _ in range(_FIX_ROUNDS):
        fixed = (repair_header(query) if header else None) or (
            fix_typos(query, now.issues, schema) if typos else None
        )
        if fixed is None:
            break
        query, now = fixed, await validate(fixed)
        if now.valid:
            # Nothing was asked of the LLM, so it cost no tokens (a zero, not an unknown)
            return [
                attempt,
                Attempt(query, now, scope, Usage(0, 0), auto_fixed=True),
            ], now
    return [attempt], validation


def _with_schema_note(
    rounds: list[Round], attempts: Sequence[Attempt], scope: SchemaScope
) -> list[Round]:
    """``rounds``, with the last one marked to tell the LLM that the schema is now the
    whole graph, the first time that is so after queries written with a part of it."""
    if (
        scope == "full"
        and rounds
        and not any(r.schema_note for r in rounds)
        and any(a.schema_scope == "filtered" and a.query for a in attempts)
    ):
        return [*rounds[:-1], replace(rounds[-1], schema_note=True)]
    return rounds
