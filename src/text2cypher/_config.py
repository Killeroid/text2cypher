"""The limits of the correction loop."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    """How many times a query may be corrected, and whether the result is executed.

    The first query is not a correction. ``attempts_w_filtered`` corrections are made
    with the filtered schema; the rest, up to ``total_attempts``, with the full one.

    With ``execute``, a query that passed validation is run (read-only) and its rows
    are on the result. If FalkorDB rejects or fails it (a type error on a value, a
    division by zero), that error is sent back for a correction, which counts like any
    other, and the result is ``failed`` when the corrections run out. Generated queries
    can be expensive, so a run is stopped after ``execute_timeout_ms`` milliseconds;
    ``None`` lets it run without limit. A timeout or a lost connection ends the run with
    the valid query and the error, because it says nothing about the query.

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
    reuse. The query stays one string: ``Result.query`` runs as it is. FalkorDB wants the
    pairs separated by spaces and a ``CYPHER`` in front, so a header with commas, or
    without it, is written that way without asking the LLM again (that changes nothing about
    what the query means) and the query is validated again.

    With ``autofix_typos``, a label, relationship type or property that the validators
    report as unknown is first replaced, without the LLM, by the one name of the graph it
    is a likely typo of (``Persn`` for ``Person``), and the query is validated again. If it
    passes, that is the result and no LLM call and no correction was spent. If it does
    not, the LLM gets the query it wrote, as usual. Off by default: a fix can make a query
    valid and still not what the question meant.

    With ``use_skills`` the LLM is offered short guides to parts of FalkorDB's Cypher that
    are easy to get wrong (graph algorithms, path finding, functions). It sees what each is
    for, and reads a skill, by calling a tool, only when the question needs it; each read
    is another call to the model, and a later correction already has what was read. A client
    that cannot be offered tools is given every skill in the prompt instead. Which skills,
    the ones that ship or your own, is the ``skills`` argument of ``Text2Cypher``.

    The schema shows up to ``example_values`` values of each property (``title: STRING (e.g.
    'Alien', 'Heat')``), so the LLM writes a value the way the graph does: spelling, case,
    format. They are read from the graph, one small query per property, and are **data from
    your graph that goes to the LLM provider**: pass ``0`` to send none. They are kept while the
    structure of the graph is the same, so a graph pays for the queries once, not on every run.

    The schema is read from the graph on every run by default. With
    ``schema_refresh_seconds`` it is kept: a run that comes after that many seconds first
    checks whether the graph changed (a few cheap queries) and reads the schema again only
    if it did. ``0`` checks on every run. Between checks a graph that changed is not seen
    in the prompt, but a query that uses a new name is still validated against the graph.

    Attributes:
        attempts_w_filtered: Corrections made with the filtered schema.
        total_attempts: Corrections in all, with the filtered and the full schema. The
            first query is not a correction.
        execute: Run a query that is valid, read-only, correct it if FalkorDB fails it, and
            put the rows on the result.
        execute_timeout_ms: Stop an execution after this many milliseconds, or never
            if ``None``.
        use_parameters: Ask the LLM for the values in a ``CYPHER`` header and parameters.
        check_types: Report type mismatches of properties.
        check_performance: Report queries that may be expensive, which is off by default.
        example_values: How many example values of each property to show the LLM; ``0``
            sends none.
        use_skills: Offer the LLM the skills, to read when a question needs them.
        autofix_typos: Fix a misspelled name in a query without the LLM when exactly one
            name of the graph is a likely typo of it.
        schema_refresh_seconds: Keep the schema of a graph and check it for changes at
            most this often, or read it on every run if ``None``.
    """

    attempts_w_filtered: int = 5
    total_attempts: int = 10
    execute: bool = False
    execute_timeout_ms: int | None = 10_000
    schema_refresh_seconds: float | None = None
    use_parameters: bool = False
    check_types: bool = True
    check_performance: bool = False
    example_values: int = 3
    autofix_typos: bool = False
    use_skills: bool = False

    def __post_init__(self) -> None:
        if self.attempts_w_filtered < 0:
            raise ValueError("attempts_w_filtered must be 0 or more")
        if self.total_attempts < 0:
            raise ValueError("total_attempts must be 0 or more")
        if self.total_attempts < self.attempts_w_filtered:
            raise ValueError("total_attempts must be at least attempts_w_filtered")
        if self.execute_timeout_ms is not None and self.execute_timeout_ms <= 0:
            raise ValueError("execute_timeout_ms must be positive")
        if self.example_values < 0:
            raise ValueError("example_values must be 0 or more")
        if self.schema_refresh_seconds is not None and self.schema_refresh_seconds < 0:
            raise ValueError("schema_refresh_seconds must be 0 or more")
