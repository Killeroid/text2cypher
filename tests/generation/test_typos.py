import pytest
from cypher_guardrail import issues as cg
from support import FILM_SCHEMA, make_issue

from text2cypher import GraphSchema
from text2cypher._generation import fix_typos
from text2cypher.schema import Path

SCHEMA = GraphSchema(
    node_properties={
        "Movie": {"title": ("String",), "released": ("Integer",)},
        "Person": {"name": ("String",), "born": ("Integer",)},
    },
    relationship_properties={"ACTED_IN": {"role": ("String",)}},
    paths=(Path("Person", "ACTED_IN", "Movie"),),
)


def label(name: str):
    return make_issue(cg.UNKNOWN_LABEL, "", label=name)


def rel_type(name: str):
    return make_issue(cg.UNKNOWN_REL_TYPE, "", relationship_type=name)


def prop(name: str):
    return make_issue(cg.UNKNOWN_PROPERTY, "", property=name)


@pytest.mark.parametrize(
    ("query", "issues", "fixed"),
    [
        (
            "MATCH (p:Persn) RETURN p.name",
            [label("Persn")],
            "MATCH (p:Person) RETURN p.name",
        ),
        (
            "MATCH (:Person)-[:ACTD_IN]->(m) RETURN m",
            [rel_type("ACTD_IN")],
            "MATCH (:Person)-[:ACTED_IN]->(m) RETURN m",
        ),
        (
            "MATCH (p:Person) RETURN p.nme",
            [prop("nme")],
            "MATCH (p:Person) RETURN p.name",
        ),
        (
            "MATCH (p:Person {nme: 'Ann'}) RETURN p",
            [prop("nme")],
            "MATCH (p:Person {name: 'Ann'}) RETURN p",
        ),
        (
            "MATCH (p:Person) WHERE p.born > 1 AND p.nme = 'x' RETURN p.nme",
            [prop("nme")],
            "MATCH (p:Person) WHERE p.born > 1 AND p.name = 'x' RETURN p.name",
        ),
        (
            "MATCH (m:Movie)<-[:ACTED_IN|ACTD_IN]-(p) RETURN m",
            [rel_type("ACTD_IN")],
            "MATCH (m:Movie)<-[:ACTED_IN|ACTED_IN]-(p) RETURN m",
        ),
        (
            "MATCH (p:person) RETURN p",
            [label("person")],
            "MATCH (p:Person) RETURN p",
        ),
        (
            "MATCH (p:Persn:Movi) RETURN p",
            [label("Persn"), label("Movi")],
            "MATCH (p:Person:Movie) RETURN p",
        ),
        (
            "MATCH (p:`Persn`) RETURN p",
            [label("Persn")],
            "MATCH (p:Person) RETURN p",
        ),
    ],
)
def test_a_misspelled_name_is_replaced_by_the_name_it_is_a_typo_of(
    query: str, issues: list, fixed: str
):
    assert fix_typos(query, issues, SCHEMA) == fixed


def test_a_name_inside_a_string_or_a_comment_is_left_alone():
    query = "MATCH (p:Persn) WHERE p.nme = 'Persn nme :Persn' RETURN p.nme // :Persn"

    assert fix_typos(query, [label("Persn"), prop("nme")], SCHEMA) == (
        "MATCH (p:Person) WHERE p.name = 'Persn nme :Persn' RETURN p.name // :Persn"
    )


def test_a_name_that_is_a_typo_of_more_than_one_name_is_not_fixed():
    schema = GraphSchema({"Person": {}, "Persons": {}}, {}, ())

    assert fix_typos("MATCH (p:Persn) RETURN p", [label("Persn")], schema) is None


def test_a_name_that_is_not_like_any_name_is_not_fixed():
    assert fix_typos("MATCH (r:Robot) RETURN r", [label("Robot")], SCHEMA) is None


def test_a_property_that_exists_on_another_label_is_not_touched():
    # which variable it belongs to is not known, so only the validators' word about it counts
    access = make_issue(
        cg.INVALID_PROPERTY_ACCESS,
        "",
        kind="node",
        name="Movie",
        properties=["name"],
    )

    assert fix_typos("MATCH (m:Movie) RETURN m.name", [access], SCHEMA) is None


def test_a_function_call_that_looks_like_a_property_is_not_renamed():
    assert fix_typos("RETURN x.nme(1)", [prop("nme")], SCHEMA) is None


def test_nothing_changes_when_no_name_is_misspelled():
    assert fix_typos("MATCH (p:Person) RETURN p", [], SCHEMA) is None
    assert fix_typos("MATCH (p:Person) RETURN p", [label("Persn")], SCHEMA) is None


def test_the_new_name_is_quoted_when_it_needs_it():
    schema = GraphSchema({"My Label": {}}, {}, ())

    assert fix_typos("MATCH (n:`My Labl`) RETURN n", [label("My Labl")], schema) == (
        "MATCH (n:`My Label`) RETURN n"
    )


def test_the_film_schema_of_the_other_tests_has_the_same_typos_fixed():
    assert (
        fix_typos("MATCH (p:Persn) RETURN p", [label("Persn")], FILM_SCHEMA)
        == "MATCH (p:Person) RETURN p"
    )
