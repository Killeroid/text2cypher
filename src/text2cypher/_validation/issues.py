"""The problems a validator found in a query, and how overlapping ones are merged."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from cypher_guardrail.issues import (
    INVALID_PROPERTY_ACCESS,
    UNKNOWN_LABEL,
    UNKNOWN_REL_TYPE,
)
from cypher_guardrail.issues import Issue as GuardrailIssue

type Source = Literal["syntax", "schema", "properties"]


@dataclass(frozen=True, slots=True)
class Issue:
    """A problem found by one of the cypher-guardrail validators.

    ``raw`` is cypher-guardrail's own typed issue: read its structured fields
    (``label``, ``suggested_path``, ...) and never parse ``description``. It is a dict,
    so it is left out of the hash (equal issues still have equal hashes).

    Attributes:
        source: The validator that found it: ``"syntax"``, ``"schema"`` or ``"properties"``.
        raw: cypher-guardrail's issue: a ``code``, a ``description`` and fields that depend
            on the code.
    """

    source: Source
    raw: GuardrailIssue = field(hash=False)

    @property
    def fields(self) -> Mapping[str, Any]:
        """``raw`` as a plain mapping, for reading the fields that depend on the code."""
        return cast("Mapping[str, Any]", self.raw)

    @property
    def code(self) -> str:
        """The issue's code, e.g. ``cypher_guardrail.Warning.UnknownLabel``."""
        return self.raw["code"]

    @property
    def description(self) -> str:
        """The issue in words, for people; read the fields of ``raw`` instead of parsing it."""
        return self.raw["description"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The outcome of validating one query against a graph.

    A score is ``None`` when it carries no information: the properties score when
    the query reads no property, and either score when the syntax is invalid and
    that validator found nothing. Validity never depends on the scores.

    Attributes:
        issues: Every issue found, in the order syntax, schema, properties.
        syntax_ok: Whether FalkorDB accepts the query.
        schema_score: The schema validator's score, 0 to 1, or ``None``.
        properties_score: The properties validator's score, 0 to 1, or ``None``.
    """

    issues: tuple[Issue, ...]
    syntax_ok: bool
    schema_score: float | None
    properties_score: float | None

    @property
    def valid(self) -> bool:
        """Whether the syntax is fine and no validator reported an issue."""
        return self.syntax_ok and not self.issues


def drop_repeated_property_issues(issues: list[Issue]) -> list[Issue]:
    """Drop property issues about a label or relationship type already reported unknown.

    cypher-guardrail reports a property of an unknown label, or of an unknown
    relationship type, as missing too, so one root cause would otherwise appear twice.
    A property issue says whether its name is a node label or a relationship type, so a
    label and a relationship type that share a name are not mixed up.
    """
    unknown = {
        ("node", issue.raw["label"])
        for issue in issues
        if issue.raw["code"] == UNKNOWN_LABEL
    } | {
        ("relationship", issue.raw["relationship_type"])
        for issue in issues
        if issue.raw["code"] == UNKNOWN_REL_TYPE
    }
    return [
        issue
        for issue in issues
        if not (
            issue.raw["code"] == INVALID_PROPERTY_ACCESS
            and (issue.raw["kind"], issue.raw["name"]) in unknown
        )
    ]
