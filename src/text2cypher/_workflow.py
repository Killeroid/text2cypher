"""The loop: generate a query, validate it, correct it, and escalate the schema."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace

from redis.exceptions import RedisError

from text2cypher._config import Config
from text2cypher._execution import Execution
from text2cypher._generation import (
    AsyncSchemaFilter,
    Generation,
    Round,
    SkillLoopFailed,
    apply_filter,
    fix_typos,
    generate_cypher,
    repair_header,
)
from text2cypher._results import (
    NO_TOKENS,
    Attempt,
    DatabaseError,
    LLMError,
    Result,
    SchemaScope,
    Status,
)
from text2cypher._skills import Skill
from text2cypher._validation import ValidationResult
from text2cypher.llm import AsyncChatClient, AsyncLLMClient, Usage
from text2cypher.schema import GraphSchema, SchemaFilter

# A wrong label stops the validators from checking its properties, so a query with both can
# need a second round of fixes; a third is for the rare case of another layer. The header of a
# query with parameters is repaired first, because a syntax error hides every other problem
_FIX_ROUNDS = 3
# A reply with no text at all is a call that returned nothing: it is made again once
_EMPTY_RETRIES = 1
# Why an empty reply is not asked for again: the same request would hit the same limit or filter
_NOT_RETRIED = {"length", "content_filter"}


async def run_workflow(
    question: str,
    schema: GraphSchema,
    *,
    llm: AsyncChatClient | AsyncLLMClient,
    validate: Callable[[str], Awaitable[ValidationResult]],
    config: Config,
    schema_filter: SchemaFilter | AsyncSchemaFilter | None = None,
    execute: Callable[[str], Awaitable[Execution]] | None = None,
    skills: Mapping[str, Skill] | None = None,
) -> Result:
    """Answer ``question`` with a query that passes ``validate``, if one can be found.

    The first query and the first ``config.attempts_w_filtered`` corrections use the
    filtered schema, the later corrections the full one, until ``config.total_attempts``
    corrections have been made. Without a ``schema_filter``, or when it keeps the whole
    schema, there is no filtered schema and every attempt uses the full one. A filter may
    be asynchronous and ask a model; what that cost is on the result as ``filter_usage``.

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

    With ``config.use_skills``, ``skills`` is what the LLM can read, by id; the skills that
    ship unless given.
    """
    filtered, filter_usage = None, NO_TOKENS
    if schema_filter is not None:
        applied = await apply_filter(schema_filter, schema, question)
        filtered = None if applied.schema == schema else applied.schema
        filter_usage = applied.usage
    attempts: list[Attempt] = []
    # The queries written so far that failed, which the next request corrects
    rounds: list[Round] = []
    corrections = 0
    # The LLM declined on the filtered schema, which is no longer used
    escalated = False
    # Replies with no text at all, in a row: the same request is made once more
    empties = 0

    while True:
        scope, used = _choose_schema(
            schema, filtered, config, corrections=corrections, escalated=escalated
        )
        rounds = _with_schema_note(rounds, attempts, scope)
        generation = await _write(
            llm, used, question, rounds, scope, attempts, filter_usage, config, skills
        )
        empties = empties + 1 if generation.empty else 0
        if not generation.query:
            ended = _no_query(
                question, generation, scope, attempts, empties, filter_usage
            )
            if ended is not None:
                return ended
            # A decline on the filtered schema is asked again on the full one; a reply
            # with no text at all is asked again as it was
            escalated = escalated or not generation.empty
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
            unchecked = Attempt(
                generation.query,
                None,
                scope,
                generation.usage,
                skills=generation.skills,
                finish_reason=generation.finish_reason,
            )
            raise DatabaseError.after(
                question, [*attempts, unchecked], error, filter_usage
            ) from error
        query = checked[-1].query
        ran = Execution()
        if validation.valid and execute is not None:
            ran = await execute(query)
        attempts.extend(
            (*checked[:-1], replace(checked[-1], execution_error=ran.error))
        )
        if validation.valid and not ran.correctable:
            return Result(
                question, "valid", tuple(attempts), ran.records, ran.error, filter_usage
            )
        if corrections == config.total_attempts:
            return Result(
                question, "failed", tuple(attempts), None, ran.error, filter_usage
            )
        corrections += 1
        rounds.append(
            Round(query, validation.issues, ran.error, skills_read=generation.skills)
        )


def _no_query(
    question: str,
    generation: Generation,
    scope: SchemaScope,
    attempts: list[Attempt],
    empties: int,
    filter_usage: Usage | None,
) -> Result | None:
    """Record a reply that has no query, and the result if it ends the run.

    An explicit empty block is the LLM declining: on the full schema that is
    ``unanswerable``. A reply with no text at all is a call that returned nothing, and
    the second one in a row is ``failed``. Otherwise the request is asked again.
    """
    attempts.append(
        Attempt(
            "",
            None,
            scope,
            generation.usage,
            skills=generation.skills,
            empty_reply=generation.empty,
            finish_reason=generation.finish_reason,
        )
    )
    status: Status = "failed" if generation.empty else "unanswerable"
    ended = (
        empties > _EMPTY_RETRIES or generation.finish_reason in _NOT_RETRIED
        if generation.empty
        else scope == "full"
    )
    if not ended:
        return None
    return Result(question, status, tuple(attempts), filter_usage=filter_usage)


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
    attempt = Attempt(
        generation.query,
        validation,
        scope,
        generation.usage,
        skills=generation.skills,
        finish_reason=generation.finish_reason,
    )
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


async def _write(
    llm: AsyncChatClient | AsyncLLMClient,
    schema: GraphSchema,
    question: str,
    rounds: Sequence[Round],
    scope: SchemaScope,
    attempts: Sequence[Attempt],
    filter_usage: Usage | None,
    config: Config,
    catalog: Mapping[str, Skill] | None,
) -> Generation:
    """The next query, or an ``LLMError`` that keeps what was paid for.

    A call that fails after others of the same request succeeded (the model had read
    skills) leaves an attempt with no query, which holds their tokens.
    """
    try:
        return await generate_cypher(
            llm,
            schema,
            question,
            rounds,
            parameterize=config.use_parameters,
            skills=config.use_skills,
            catalog=catalog,
        )
    except SkillLoopFailed as failed:
        partial = [Attempt("", None, scope, failed.usage)] if failed.calls else []
        cause = failed.__cause__ or failed
        raise LLMError.after(
            question, [*attempts, *partial], cause, filter_usage
        ) from cause
    except Exception as error:
        raise LLMError.after(question, attempts, error, filter_usage) from error


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
