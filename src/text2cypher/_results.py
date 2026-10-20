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
    """

    query: str
    validation: ValidationResult | None
    schema_scope: SchemaScope
    usage: Usage | None


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

    ``valid``: the last query passed validation. ``unanswerable``: the LLM declined,
    so there is no query. ``failed``: the corrections ran out; the last attempt has
    the last query and its issues.

    When execution is on and the status is ``valid``, ``records`` has what the query
    returned, or ``execution_error`` says why FalkorDB did not run it (the status
    stays ``valid``: the query passed validation).

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
