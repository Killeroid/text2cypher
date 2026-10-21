"""What came of one question: the result, its attempts and what a query returned."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Self

from text2cypher._errors import Text2CypherError
from text2cypher._validation import ValidationResult
from text2cypher.llm import Usage

type SchemaScope = Literal["filtered", "full"]
type Status = Literal["valid", "unanswerable", "failed"]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One query the LLM wrote, how it was validated, and what writing it cost.

    ``validation`` is ``None`` when the query is empty (the LLM declined), which is
    not validated, and on the last attempt of an ``LLMError`` or ``DatabaseError``
    when the database failed while the query was being validated.

    Attributes:
        query: The query, empty if the LLM declined.
        validation: How the query was validated, or ``None`` for an empty query.
        schema_scope: The schema the query was written for: ``"filtered"`` or ``"full"``.
        usage: The tokens the call that wrote the query used, if the client reported them.
        execution_error: What FalkorDB said when it ran the query and failed, if execution is
            on and the query was run; the next attempt then corrects it.
        auto_fixed: The query is the one before it with misspelled names replaced, without
            the LLM (``Config.autofix_typos``). It cost no tokens, and it was not written by
            a request, so it is not a correction.
    """

    query: str
    validation: ValidationResult | None
    schema_scope: SchemaScope
    usage: Usage | None
    execution_error: str | None = None
    auto_fixed: bool = False


def total_usage(attempts: Iterable[Attempt]) -> Usage | None:
    """The tokens of ``attempts``, or ``None`` if any of them did not report them."""
    total = Usage(0, 0)
    for attempt in attempts:
        if attempt.usage is None:
            return None
        total += attempt.usage
    return total


@dataclass(frozen=True, slots=True)
class Records:
    """What a query returned: the names of its columns, and its rows.

    Values are the client's own: nodes, relationships and paths are ``falkordb``
    objects.

    Attributes:
        columns: The names of the columns, as the query returns them.
        rows: The rows, one tuple of values per row.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True, slots=True)
class Result:
    """What came of one question: every attempt, in order, and how it ended.

    ``valid``: the last query passed validation (and ran, if execution is on).
    ``unanswerable``: the LLM declined, so there is no query. ``failed``: the corrections
    ran out; the last attempt has the last query and its issues, or the error FalkorDB
    gave when it ran it.

    When execution is on, a query that FalkorDB rejects or fails is corrected like one
    that fails validation. The status is ``valid`` with ``records`` when a query runs,
    and with ``execution_error`` when a timeout or a lost connection stops it (that says
    nothing about the query). It is ``failed`` when the corrections run out on queries
    that FalkorDB fails.

    A result is hashable unless it holds ``records``, whose values are the client's own
    objects.

    Attributes:
        question: The question that was asked.
        status: How it ended: ``"valid"``, ``"unanswerable"`` or ``"failed"``.
        attempts: Every query the LLM wrote, in order, with its validation, schema and usage.
        records: What the final query returned, when it was executed and ran.
        execution_error: Why the final query did not run, when it was executed and failed.
    """

    question: str
    status: Status
    attempts: tuple[Attempt, ...]
    records: Records | None = None
    execution_error: str | None = None

    @property
    def query(self) -> str:
        """The last query, empty if the LLM declined."""
        return self.attempts[-1].query

    @property
    def usage(self) -> Usage | None:
        """The tokens of all attempts, or ``None`` if any attempt did not report them."""
        return total_usage(self.attempts)


class RunInterrupted(Text2CypherError):
    """A run that stopped in the middle, with what had been done and paid for.

    The LLM calls before the failure were paid for, so they are kept here as they would be
    on a ``Result``. The error that stopped the run is ``__cause__``.

    Attributes:
        question: The question that was asked.
        attempts: The attempts that were made before the failure, in order. For a
            ``DatabaseError`` the last one is the query that was being validated, with no
            validation.
        usage: The tokens of those attempts, or ``None`` if any did not report them. The
            call that failed is not counted: nothing says what it cost.
    """

    what = "The run stopped"

    def __init__(
        self, question: str, attempts: Iterable[Attempt], message: str
    ) -> None:
        super().__init__(message)
        self.question = question
        self.attempts = tuple(attempts)
        self.usage = total_usage(self.attempts)

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.question, self.attempts, str(self))

    @classmethod
    def after(
        cls, question: str, attempts: Iterable[Attempt], cause: BaseException
    ) -> Self:
        """The error for ``cause`` after ``attempts``; raise it ``from`` the cause."""
        attempts = tuple(attempts)
        return cls(
            question,
            attempts,
            f"{cls.what} after {len(attempts)} attempt(s) at {question!r}: {cause}",
        )


class LLMError(RunInterrupted):
    """The LLM client raised while a question was being answered.

    Any exception the client raises (a rate limit, a lost connection, a bad key) ends the
    run with this one, so the attempts and the tokens before it are not lost. Cancelling
    the task is not an error and passes through unchanged.
    """

    what = "The LLM failed"


class DatabaseError(RunInterrupted):
    """FalkorDB failed while a query the LLM wrote was being validated.

    A connection that is lost or a server that fails (``redis.exceptions.RedisError``)
    ends the run with this one. A query that FalkorDB rejects when it is run is not one:
    that is corrected, or recorded as ``execution_error``.
    """

    what = "The database failed"
