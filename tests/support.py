"""Helpers shared by the tests: builders for the values the library passes around."""

from typing import Any, cast

from cypher_guardrail import issues as cg

from text2cypher import GraphSchema, Issue, Records, ValidationResult
from text2cypher._execution import Execution
from text2cypher.schema import Path

# A small schema, built by hand: (:Person {name})-[:ACTED_IN]->(:Movie {title})
FILM_SCHEMA = GraphSchema(
    node_properties={"Movie": {"title": ("String",)}, "Person": {"name": ("String",)}},
    relationship_properties={"ACTED_IN": {}},
    paths=(Path("Person", "ACTED_IN", "Movie"),),
)

VALID = ValidationResult((), True, 1.0, 1.0)


def reply(query: str) -> str:
    """What an LLM answers with: the query in a fenced block."""
    return f"```cypher\n{query}\n```"


def make_issue(
    code: str, description: str = "", source: str = "schema", **fields: Any
) -> Issue:
    """An issue with the given code and fields, as a validator would report it."""
    raw = {"code": code, "description": description, **fields}
    return Issue(source=cast("Any", source), raw=cast("Any", raw))


def invalid(description: str, code: str = cg.UNKNOWN_LABEL) -> ValidationResult:
    """A validation result with one issue (an unknown label gets the field it always has)."""
    fields = {"label": "Robot"} if code == cg.UNKNOWN_LABEL else {}
    return ValidationResult((make_issue(code, description, **fields),), True, 0.0, None)


def misspelled(
    *, label: str | None = None, type: str | None = None, prop: str | None = None
) -> ValidationResult:
    """A validation result for a query that uses a name the graph does not have."""
    issues = []
    if label:
        issues.append(make_issue(cg.UNKNOWN_LABEL, "No label.", label=label))
    if type:
        issues.append(
            make_issue(cg.UNKNOWN_REL_TYPE, "No type.", relationship_type=type)
        )
    if prop:
        issues.append(make_issue(cg.UNKNOWN_PROPERTY, "No property.", property=prop))
    return ValidationResult(tuple(issues), True, 0.0, None)


class ScriptedValidator:
    """Answers each call with the next canned result, and records the queries."""

    def __init__(self, *results: ValidationResult) -> None:
        self._results = iter(results)
        self.queries: list[str] = []

    async def __call__(self, query: str) -> ValidationResult:
        self.queries.append(query)
        return next(self._results)


ROWS = Records(("x",), ((1,),))
RAN = Execution(records=ROWS)


def failed_to_run(error: str) -> Execution:
    """FalkorDB failed the query: another query may not fail."""
    return Execution(error=error, correctable=True)


class ScriptedExecutor:
    """Answers each run with the next canned execution, and records the queries."""

    def __init__(self, *executions: Execution) -> None:
        self._executions = iter(executions)
        self.queries: list[str] = []

    async def __call__(self, query: str) -> Execution:
        self.queries.append(query)
        return next(self._executions)
