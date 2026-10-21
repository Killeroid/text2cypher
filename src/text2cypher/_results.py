"""What came of one question: the result, its attempts and what a query returned."""

from dataclasses import dataclass
from typing import Any, Literal

from text2cypher._validation import ValidationResult
from text2cypher.llm import Usage

type SchemaScope = Literal["filtered", "full"]
type Status = Literal["valid", "unanswerable", "failed"]


@dataclass(frozen=True, slots=True)
class Attempt:
    """One query the LLM wrote, how it was validated, and what writing it cost.

    ``validation`` is ``None`` when the query is empty (the LLM declined), which is
    not validated.

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
        total = Usage(0, 0)
        for attempt in self.attempts:
            if attempt.usage is None:
                return None
            total += attempt.usage
        return total
