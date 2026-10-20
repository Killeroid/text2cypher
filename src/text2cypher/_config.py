"""The limits of the correction loop."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    """How many times a query may be corrected, and whether the result is executed.

    The first query is not a correction. ``attempts_w_filtered`` corrections are made
    with the filtered schema; the rest, up to ``total_attempts``, with the full one.

    With ``execute``, a query that passed validation is run (read-only) and its rows
    are on the result. Generated queries can be expensive, so a run is stopped after
    ``execute_timeout_ms`` milliseconds; ``None`` lets it run without limit.

    Two optional checks report more than the validators do by default. ``check_types``
    (on) reports a property compared with a value of another type, and ``sum``, ``avg`` and
    the like of a property that holds no numbers: such a query fails or returns nothing.
    ``check_performance`` (off) reports a cartesian product, a scan of every node and a path
    with no upper bound; each can be what the question needs (a shortest path between two
    nodes is a cartesian product by design). Every issue a check reports is corrected like
    any other.

    By default the LLM writes the values of a query inside it. With ``use_parameters`` it
    puts them in a ``CYPHER`` header before the query (``CYPHER name='Ann' MATCH (p
    {name: $name}) ...``) and uses them as parameters, which FalkorDB can plan once and
    reuse. The query stays one string: ``Result.query`` runs as it is.

    The schema is read from the graph on every run by default. With
    ``schema_refresh_seconds`` it is kept: a run that comes after that many seconds first
    checks whether the graph changed (a few cheap queries) and reads the schema again only
    if it did. ``0`` checks on every run. Between checks a graph that changed is not seen
    in the prompt, but a query that uses a new name is still validated against the graph.

    Attributes:
        attempts_w_filtered: Corrections made with the filtered schema.
        total_attempts: Corrections in all, with the filtered and the full schema. The
            first query is not a correction.
        execute: Run the final query when it is valid, read-only, and put the rows on
            the result.
        execute_timeout_ms: Stop an execution after this many milliseconds, or never
            if ``None``.
        use_parameters: Ask the LLM for the values in a ``CYPHER`` header and parameters.
        check_types: Report type mismatches of properties.
        check_performance: Report queries that may be expensive, which is off by default.
        schema_refresh_seconds: Keep the schema of a graph and check it for changes at
            most this often, or read it on every run if ``None``.
    """

    attempts_w_filtered: int = 3
    total_attempts: int = 5
    execute: bool = False
    execute_timeout_ms: int | None = 10_000
    schema_refresh_seconds: float | None = None
    use_parameters: bool = False
    check_types: bool = True
    check_performance: bool = False

    def __post_init__(self) -> None:
        if self.attempts_w_filtered < 0:
            raise ValueError("attempts_w_filtered must be 0 or more")
        if self.total_attempts < 0:
            raise ValueError("total_attempts must be 0 or more")
        if self.total_attempts < self.attempts_w_filtered:
            raise ValueError("total_attempts must be at least attempts_w_filtered")
        if self.execute_timeout_ms is not None and self.execute_timeout_ms <= 0:
            raise ValueError("execute_timeout_ms must be positive")
        if self.schema_refresh_seconds is not None and self.schema_refresh_seconds < 0:
            raise ValueError("schema_refresh_seconds must be 0 or more")
