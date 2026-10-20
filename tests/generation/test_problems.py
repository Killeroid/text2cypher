"""How the problems of a query are put to the LLM, from the fields of each issue."""

from cypher_guardrail import issues as cg
from support import make_issue

from text2cypher import GraphSchema
from text2cypher._generation.problems import problems
from text2cypher.schema import Path

STRING = ("String",)
SCHEMA = GraphSchema(
    node_properties={
        "Employee": {"name": STRING, "salary": STRING},
        "Department": {"name": STRING},
        "Supplier": {"name": STRING, "rating": STRING},
        "Task": {"title": STRING},
        "Order": {"number": STRING},
        "OrderLine": {"quantity": STRING},
        "Office": {"name": STRING},
        "City": {"name": STRING},
        "Customer": {"name": STRING},
        "Country": {"name": STRING},
    },
    relationship_properties={
        "WORKS_IN": {},
        "MANAGES": {},
        "ASSIGNED_TO": {"hours": STRING},
        "LOCATED_IN": {},
    },
    paths=(
        Path("Employee", "WORKS_IN", "Department"),
        Path("Employee", "MANAGES", "Department"),
        Path("Employee", "ASSIGNED_TO", "Task"),
        Path("Office", "LOCATED_IN", "City"),
        Path("Customer", "LOCATED_IN", "Country"),
    ),
)


def test_an_unknown_label_is_said_plainly_with_the_closest_name_as_a_hint():
    text = problems(
        [make_issue(cg.UNKNOWN_LABEL, "long FalkorDB text", label="Employe")], SCHEMA
    )

    assert (
        text
        == "- There is no label `Employe` in the schema.\n  Did you mean `Employee`?"
    )


def test_an_unknown_label_with_no_close_name_has_no_hint():
    text = problems([make_issue(cg.UNKNOWN_LABEL, "x", label="Vendor")], SCHEMA)

    assert text == "- There is no label `Vendor` in the schema."


def test_an_unknown_relationship_type_has_a_hint_too():
    text = problems(
        [make_issue(cg.UNKNOWN_REL_TYPE, "x", relationship_type="MANAGED")], SCHEMA
    )

    assert text == (
        "- There is no relationship type `MANAGED` in the schema.\n"
        "  Did you mean `MANAGES`?"
    )


def test_a_property_of_a_relationship_read_from_a_node_says_where_it_is():
    text = problems(
        [
            make_issue(
                cg.INVALID_PROPERTY_ACCESS,
                "x",
                source="properties",
                kind="node",
                name="Task",
                properties=["hours"],
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The label `Task` has no property `hours`.\n"
        "  `hours` is a property of the relationship type `ASSIGNED_TO`.\n"
        "  `Task` has: title."
    )


def test_a_property_of_another_label_says_where_it_is():
    text = problems(
        [
            make_issue(
                cg.INVALID_PROPERTY_ACCESS,
                "x",
                source="properties",
                kind="node",
                name="Department",
                properties=["salary"],
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The label `Department` has no property `salary`.\n"
        "  `salary` is a property of the label `Employee`.\n"
        "  `Department` has: name."
    )


def test_a_property_that_is_nowhere_hints_at_the_closest_property_of_the_label():
    text = problems(
        [
            make_issue(
                cg.INVALID_PROPERTY_ACCESS,
                "x",
                source="properties",
                kind="node",
                name="Supplier",
                properties=["rate"],
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The label `Supplier` has no property `rate`.\n  Did you mean `rating`?"
    )


def test_a_property_that_is_nowhere_and_close_to_nothing_lists_the_ones_there_are():
    text = problems(
        [
            make_issue(
                cg.INVALID_PROPERTY_ACCESS,
                "x",
                source="properties",
                kind="node",
                name="Country",
                properties=["colour"],
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The label `Country` has no property `colour`.\n  `Country` has: name."
    )


def test_an_unknown_property_key_is_dropped_when_the_label_issue_says_more():
    unknown = make_issue(cg.UNKNOWN_PROPERTY, "x", source="properties", property="rate")
    invalid = make_issue(
        cg.INVALID_PROPERTY_ACCESS,
        "x",
        source="properties",
        kind="node",
        name="Supplier",
        properties=["rate"],
    )

    text = problems([unknown, invalid], SCHEMA)

    assert text == (
        "- The label `Supplier` has no property `rate`.\n  Did you mean `rating`?"
    )


def test_an_unknown_property_key_hints_at_the_closest_property_and_its_owner():
    text = problems(
        [make_issue(cg.UNKNOWN_PROPERTY, "x", source="properties", property="rate")],
        SCHEMA,
    )

    assert text == (
        "- No label or relationship type has a property `rate`.\n"
        "  Did you mean `rating` (a property of `Supplier`)?"
    )


def test_a_path_that_does_not_exist_lists_where_its_relationship_type_does():
    text = problems(
        [
            make_issue(
                cg.UNKNOWN_PATH,
                "x",
                path="(:Office)-[:LOCATED_IN]->(:Country)",
                original_path="(o:Office)-[:LOCATED_IN]->(:Country {name: 'Canada'})",
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The pattern (:Office)-[:LOCATED_IN]->(:Country) does not exist in the schema.\n"
        "  `LOCATED_IN` goes between: (:Office)-[:LOCATED_IN]->(:City), "
        "(:Customer)-[:LOCATED_IN]->(:Country)"
    )


def test_a_path_with_a_name_in_backticks_still_finds_its_relationship_type():
    schema = GraphSchema(
        node_properties={"A": {}, "B": {}, "C": {}},
        relationship_properties={"HAS A SPACE": {}},
        paths=(Path("A", "HAS A SPACE", "B"),),
    )

    text = problems(
        [
            make_issue(
                cg.UNKNOWN_PATH,
                "x",
                path="(:A)-[:`HAS A SPACE`]->(:C)",
                original_path="",
            )
        ],
        schema,
    )

    assert "(:A)-[:`HAS A SPACE`]->(:B)" in text


def test_a_path_of_an_unknown_shape_falls_back_to_the_description():
    text = problems(
        [make_issue(cg.UNKNOWN_PATH, "The path is wrong.", path="???")], SCHEMA
    )

    assert text == "- The path is wrong."


def test_other_issues_keep_their_description_and_suggested_path():
    text = problems(
        [
            make_issue(
                cg.WRONG_DIRECTION_PATH,
                "The direction is the other way.",
                suggested_path="(a)<-[:R]-(b)",
            )
        ],
        SCHEMA,
    )

    assert text == "- The direction is the other way.\n  Suggested: (a)<-[:R]-(b)"


def test_a_write_query_is_worded_for_the_model():
    text = problems(
        [make_issue(cg.WRITE_QUERY_ERROR, "Create with read_only=False")], SCHEMA
    )

    assert text == "- The query can modify the graph. Only read queries are allowed."


def test_a_problem_that_is_repeated_is_listed_once():
    same = make_issue(cg.UNKNOWN_LABEL, "x", label="Vendor")

    assert problems([same, same], SCHEMA).count("Vendor") == 1


def test_a_name_that_only_shares_letters_with_another_is_not_offered_as_a_correction():
    schema = GraphSchema(
        node_properties={"City": {}}, relationship_properties={}, paths=()
    )

    text = problems([make_issue(cg.UNKNOWN_LABEL, "x", label="Client")], schema)

    assert text == "- There is no label `Client` in the schema."


def test_a_parameter_without_a_value_is_told_to_be_written_in_the_query():
    text = problems(
        [
            make_issue(
                cg.PARAMETER_NOT_PROVIDED,
                "Did not supply query with enough parameters.",
                source="syntax",
                parameters=["name", "year"],
            )
        ],
        SCHEMA,
    )

    assert text == (
        "- The query uses the parameters `$name`, `$year`, which have no value. "
        "Write the value in the query instead."
    )


def test_a_single_parameter_is_worded_in_the_singular():
    text = problems(
        [make_issue(cg.PARAMETER_NOT_PROVIDED, "x", source="syntax", parameters=["n"])],
        SCHEMA,
    )

    assert text.startswith("- The query uses the parameter `$n`, which has no value.")
