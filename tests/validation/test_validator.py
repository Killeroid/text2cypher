import asyncio
from collections import Counter
from typing import TYPE_CHECKING, Any, cast

import pytest
from cypher_guardrail import Snapshot
from cypher_guardrail import issues as cg
from falkordb.asyncio import FalkorDB

from text2cypher import GraphNotFoundError
from text2cypher._validation import QueryValidator

if TYPE_CHECKING:
    from collections.abc import Mapping


pytestmark = pytest.mark.anyio

# Schema of the ``movies_graph`` fixture:
#   (:Person {name})-[:ACTED_IN {role}]->(:Movie {title, year})

# The structured fields we rely on, per issue kind: (query, source, code, fields).
# Doubles as the contract test for cypher_guardrail.issues: if it renames a field
# or changes what it reports, these fail instead of the library degrading silently.
FIELD_CASES: dict[str, tuple[str, str, str, dict[str, Any]]] = {
    "unknown label": (
        "MATCH (p:Robot) RETURN p.name",
        "schema",
        cg.UNKNOWN_LABEL,
        {"label": "Robot"},
    ),
    "unknown relationship type": (
        "MATCH (p:Person)-[:LOVES]->(m:Movie) RETURN p.name",
        "schema",
        cg.UNKNOWN_REL_TYPE,
        {"relationship_type": "LOVES"},
    ),
    "invalid property access": (
        "MATCH (p:Person) RETURN p.title",
        "properties",
        cg.INVALID_PROPERTY_ACCESS,
        {"kind": "node", "name": "Person", "properties": ["title"]},
    ),
    "invalid relationship property access": (
        "MATCH ()-[r:ACTED_IN]->() RETURN r.title",
        "properties",
        cg.INVALID_PROPERTY_ACCESS,
        {"kind": "relationship", "name": "ACTED_IN", "properties": ["title"]},
    ),
    "unknown property key": (
        "MATCH (p:Person) RETURN p.zzz",
        "properties",
        cg.UNKNOWN_PROPERTY,
        {"property": "zzz"},
    ),
    "wrong direction": (
        "MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN p.name",
        "schema",
        cg.WRONG_DIRECTION_PATH,
        {
            "path": "(:Movie)-[:ACTED_IN]->(:Person)",
            "original_path": "(m:Movie)-[:ACTED_IN]->(p:Person)",
            "suggested_path": "(m:Movie)<-[:ACTED_IN]-(p:Person)",
        },
    ),
    "variable length path": (
        "MATCH (p:Person)-[:ACTED_IN*2..3]->(m:Movie) RETURN m.title",
        "schema",
        cg.VARIABLE_LENGTH_PATH,
        {
            "path": "(:Person)-[:ACTED_IN*2..3]->(:Movie)",
            "original_path": "(p:Person)-[:ACTED_IN*2..3]->(m:Movie)",
            "suggested_path": "(p:Person)-[:ACTED_IN*]->(m:Movie)",
        },
    ),
    "unknown path": (
        "MATCH (m:Movie)-[:ACTED_IN]->(m2:Movie) RETURN m.title",
        "schema",
        cg.UNKNOWN_PATH,
        {
            "path": "(:Movie)-[:ACTED_IN]->(:Movie)",
            "original_path": "(m:Movie)-[:ACTED_IN]->(m2:Movie)",
        },
    ),
    "missing parameter": (
        "MATCH (p:Person {name: $n}) RETURN p",
        "syntax",
        cg.PARAMETER_NOT_PROVIDED,
        {"parameters": ["n"]},
    ),
    "unbound variable": (
        "MATCH (p:Person) RETURN q.name",
        "syntax",
        cg.UNBOUND_VARIABLE_ERROR,
        {"variable": "q"},
    ),
    "unknown function": (
        "MATCH (p:Person) RETURN nosuch.fn(p.name)",
        "syntax",
        cg.UNKNOWN_FUNCTION_ERROR,
        {"name": "nosuch.fn"},
    ),
    "unknown function in the instance": (
        "MATCH (p:Person) RETURN nosuch.fn(p.name)",
        "schema",
        cg.UNKNOWN_FUNCTION,
        {"name": "nosuch.fn"},
    ),
    "invalid limit": (
        "MATCH (p:Person) RETURN p.name LIMIT 'ten'",
        "syntax",
        cg.INVALID_LIMIT_ERROR,
        {"clause": "LIMIT"},
    ),
    "conflicting variables": (
        "MATCH (a:Person)-[:ACTED_IN]->(a:Movie) RETURN a",
        "syntax",
        cg.CONFLICTING_VARIABLES,
        {"variables": {"a": ["Person", "Movie"]}},
    ),
}


@pytest.fixture
def validator(falkordb_client: FalkorDB) -> QueryValidator:
    return QueryValidator(falkordb_client)


async def test_valid_query(validator: QueryValidator, movies_graph: str):
    result = await validator.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title", movies_graph
    )

    assert result.valid
    assert result.issues == ()
    assert result.syntax_ok
    assert result.schema_score == 1.0
    assert result.properties_score == 1.0


@pytest.mark.parametrize(
    ("query", "source", "code", "fields"),
    FIELD_CASES.values(),
    ids=FIELD_CASES.keys(),
)
async def test_issue_has_structured_fields(
    validator: QueryValidator,
    movies_graph: str,
    query: str,
    source: str,
    code: str,
    fields: dict[str, Any],
):
    result = await validator.validate(query, movies_graph)

    [issue] = [i for i in result.issues if i.code == code]
    assert issue.source == source
    raw = cast("Mapping[str, Any]", issue.raw)
    assert {key: raw[key] for key in fields} == fields
    assert issue.description
    assert not result.valid


async def test_syntax_error(validator: QueryValidator, movies_graph: str):
    result = await validator.validate("MATCH (p:Person RETURN p", movies_graph)

    assert not result.valid
    assert not result.syntax_ok
    assert [(i.source, i.code) for i in result.issues] == [("syntax", cg.SYNTAX_ERROR)]


async def test_scores_are_none_when_syntax_fails_and_nothing_was_found(
    validator: QueryValidator, movies_graph: str
):
    # cypher-guardrail reports schema 1 / properties None here, which means "nothing to check"
    result = await validator.validate("this is not cypher at all", movies_graph)

    assert result.schema_score is None
    assert result.properties_score is None


async def test_scores_are_none_even_if_extraction_succeeded_on_an_invalid_query(
    validator: QueryValidator, movies_graph: str
):
    result = await validator.validate("MATCH (p:Person) RETURN p.name,", movies_graph)

    assert not result.syntax_ok
    assert result.schema_score is None
    assert result.properties_score is None


async def test_other_validators_still_report_when_syntax_fails(
    validator: QueryValidator, movies_graph: str
):
    result = await validator.validate("MATCH (p:Robot) RETURN p.name, ,", movies_graph)

    assert [(i.source, i.code) for i in result.issues] == [
        ("syntax", cg.SYNTAX_ERROR),
        ("schema", cg.UNKNOWN_LABEL),
    ]
    # these validators did report something, so their scores are kept
    assert result.schema_score == 0.0


async def test_property_issue_on_unknown_label_is_not_reported_twice(
    validator: QueryValidator, movies_graph: str
):
    result = await validator.validate("MATCH (p:Robot) RETURN p.name", movies_graph)

    assert [i.code for i in result.issues] == [cg.UNKNOWN_LABEL]


async def test_property_issue_on_known_label_is_kept_next_to_unknown_label(
    validator: QueryValidator, movies_graph: str
):
    result = await validator.validate(
        "MATCH (p:Robot), (m:Movie) RETURN p.name, m.name", movies_graph
    )

    assert [(i.code, i.raw.get("name")) for i in result.issues] == [
        (cg.UNKNOWN_LABEL, None),
        (cg.INVALID_PROPERTY_ACCESS, "Movie"),
    ]


async def test_property_issue_is_kept_when_an_unknown_relationship_type_shares_its_label(
    validator: QueryValidator, movies_graph: str
):
    # :Movie is a real label but not a relationship type; m.name is wrong on its own
    result = await validator.validate(
        "MATCH (m:Movie)-[:Movie]->(p:Person) RETURN m.name", movies_graph
    )

    assert [(i.code, i.raw.get("name")) for i in result.issues] == [
        (cg.UNKNOWN_REL_TYPE, None),
        (cg.INVALID_PROPERTY_ACCESS, "Movie"),
    ]


async def test_property_issue_on_unknown_relationship_type_is_not_reported_twice(
    validator: QueryValidator, movies_graph: str
):
    result = await validator.validate(
        "MATCH ()-[r:LOVES]->() RETURN r.zzz", movies_graph
    )

    assert [i.code for i in result.issues] == [cg.UNKNOWN_REL_TYPE]


async def test_write_query_is_not_valid(validator: QueryValidator, movies_graph: str):
    result = await validator.validate("CREATE (:Person {name: 'x'})", movies_graph)

    assert not result.valid
    assert [(i.source, i.code) for i in result.issues] == [
        ("syntax", cg.WRITE_QUERY_ERROR)
    ]


async def test_other_validators_are_cancelled_when_one_raises(
    validator: QueryValidator, movies_graph: str, monkeypatch: pytest.MonkeyPatch
):
    finished = False

    async def fails(query: str, graph: str | None = None):
        await asyncio.sleep(0)
        raise RuntimeError("connection lost")

    async def slow(query: str, graph: str | None = None):
        nonlocal finished
        await asyncio.sleep(0.2)
        finished = True

    monkeypatch.setattr(validator._syntax, "validate", fails)
    monkeypatch.setattr(validator._schema, "validate", slow)

    with pytest.raises(RuntimeError, match="connection lost"):
        await validator.validate("MATCH (n) RETURN n", movies_graph)
    await asyncio.sleep(0.5)

    assert not finished


async def test_graph_can_be_given_instead_of_a_graph_name(
    falkordb_client: FalkorDB, movies_graph: str
):
    validator = QueryValidator(falkordb_client.select_graph(movies_graph))

    result = await validator.validate("MATCH (p:Person) RETURN p.name")

    assert result.valid


async def test_no_graph_raises(validator: QueryValidator):
    with pytest.raises(ValueError, match="graph name"):
        await validator.validate("MATCH (n) RETURN n")


async def test_a_graph_that_does_not_exist_raises_our_error(validator: QueryValidator):
    # The graph can be deleted after the schema was read; cypher-guardrail 1.2 raises its
    # own error for it, and callers catch ours
    with pytest.raises(GraphNotFoundError, match="no_such_graph"):
        await validator.validate("MATCH (n:Person) RETURN n", "no_such_graph")


async def test_a_snapshot_gives_the_same_answers_with_fewer_requests(
    falkordb_client: FalkorDB, movies_graph: str, commands: Counter[str]
):
    snapshot = await Snapshot.capture_async(falkordb_client, movies_graph, types=True)
    online = QueryValidator(falkordb_client)
    hybrid = QueryValidator(falkordb_client, snapshot=snapshot)
    queries = [query for query, *_ in FIELD_CASES.values()] + [
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title",
        "MATCH (p:Person RETURN p",
        "CREATE (:Person {name: 'x'})",
    ]

    for query in queries:
        commands.clear()
        expected = await online.validate(query, movies_graph)
        online_requests = sum(commands.values())
        commands.clear()
        got = await hybrid.validate(query, movies_graph)

        assert got == expected, query
        assert sum(commands.values()) <= online_requests, query

    # A query that is fine needs the database only to plan it, not to look up its names
    commands.clear()
    await hybrid.validate(
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name", movies_graph
    )
    assert sum(commands.values()) == 1


PERFORMANCE_CASES = {
    "cartesian product": (
        "MATCH (p:Person), (m:Movie) RETURN p.name, m.title",
        cg.CARTESIAN_PRODUCT,
        {"variables": [["p"], ["m"]]},
    ),
    "unlabeled scan": (
        "MATCH (n) WHERE n.name = 'Keanu' RETURN n.name",
        cg.UNLABELED_NODE_SCAN,
        {"variable": "n"},
    ),
    "unbounded path": (
        "MATCH (p:Person)-[:ACTED_IN*]->(m:Movie) RETURN p.name",
        cg.UNBOUNDED_PATH,
        {"path": "(p:Person)-[:ACTED_IN*]->(m:Movie)"},
    ),
}
TYPE_CASES = {
    "compared with another type": (
        "MATCH (p:Person) WHERE p.name = 5 RETURN p.name",
        cg.PROPERTY_TYPE_MISMATCH,
        {"kind": "node", "name": "Person", "property": "name", "value_type": "Integer"},
    ),
    "aggregate of a string": (
        "MATCH (m:Movie) RETURN sum(m.title)",
        cg.AGGREGATE_TYPE_MISMATCH,
        {"function": "sum", "kind": "node", "name": "Movie", "property": "title"},
    ),
}


@pytest.mark.parametrize(
    ("query", "code", "fields"),
    PERFORMANCE_CASES.values(),
    ids=PERFORMANCE_CASES.keys(),
)
async def test_performance_checks_are_off_unless_asked_for(
    falkordb_client: FalkorDB, movies_graph: str, query: str, code: str, fields: dict
):
    off = await QueryValidator(falkordb_client).validate(query, movies_graph)
    on = await QueryValidator(falkordb_client, check_performance=True).validate(
        query, movies_graph
    )

    assert off.valid
    [issue] = [i for i in on.issues if i.code == code]
    assert issue.source == "syntax"
    assert {key: issue.fields[key] for key in fields} == fields
    assert not on.valid  # every issue is corrected, a warning too


@pytest.mark.parametrize(
    ("query", "code", "fields"), TYPE_CASES.values(), ids=TYPE_CASES.keys()
)
async def test_type_checks_are_off_unless_asked_for(
    falkordb_client: FalkorDB, movies_graph: str, query: str, code: str, fields: dict
):
    off = await QueryValidator(falkordb_client).validate(query, movies_graph)
    on = await QueryValidator(falkordb_client, check_types=True).validate(
        query, movies_graph
    )

    assert off.valid
    [issue] = [i for i in on.issues if i.code == code]
    assert issue.source == "properties"
    assert {key: issue.fields[key] for key in fields} == fields


async def test_the_optional_checks_work_from_a_snapshot_too(
    falkordb_client: FalkorDB, movies_graph: str
):
    snapshot = await Snapshot.capture_async(falkordb_client, movies_graph, types=True)
    validator = QueryValidator(
        falkordb_client, snapshot=snapshot, check_types=True, check_performance=True
    )

    types = await validator.validate(
        "MATCH (m:Movie) RETURN sum(m.title)", movies_graph
    )
    plan = await validator.validate(
        "MATCH (p:Person), (m:Movie) RETURN p.name", movies_graph
    )

    assert [i.code for i in types.issues] == [cg.AGGREGATE_TYPE_MISMATCH]
    assert [i.code for i in plan.issues] == [cg.CARTESIAN_PRODUCT]
