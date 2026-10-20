from cypher_guardrail import issues as cg
from support import make_issue

from text2cypher._results import Attempt
from text2cypher._validation import (
    Issue,
    ValidationResult,
    drop_repeated_property_issues,
)


def test_drop_repeated_property_issues_drops_by_kind_and_name():
    unknown_label = make_issue(cg.UNKNOWN_LABEL, label="Robot")
    unknown_type = make_issue(cg.UNKNOWN_REL_TYPE, relationship_type="Movie")

    def about(kind: str, name: str) -> Issue:
        return make_issue(
            cg.INVALID_PROPERTY_ACCESS,
            source="properties",
            kind=kind,
            name=name,
            properties=["x"],
        )

    on_robot = about("node", "Robot")
    on_type = about("relationship", "Movie")
    # a real label that shares its name with the unknown relationship type
    on_movie = about("node", "Movie")

    kept = drop_repeated_property_issues(
        [unknown_label, unknown_type, on_robot, on_type, on_movie]
    )

    assert kept == [unknown_label, unknown_type, on_movie]


def test_an_issue_can_be_hashed_and_kept_in_a_set():
    robot = make_issue(cg.UNKNOWN_LABEL, "No label.", label="Robot")
    same = make_issue(cg.UNKNOWN_LABEL, "No label.", label="Robot")
    other = make_issue(cg.UNKNOWN_LABEL, "No label.", label="Alien")

    assert hash(robot) == hash(same)
    assert {robot, same, other} == {robot, other}


def test_a_validation_result_and_an_attempt_can_be_hashed():
    issue = make_issue(cg.UNKNOWN_LABEL, "No label.", label="Robot")
    validation = ValidationResult((issue,), True, 0.0, None)

    assert hash(validation) == hash(ValidationResult((issue,), True, 0.0, None))
    assert (
        len(
            {
                Attempt("q", validation, "full", None),
                Attempt("q", validation, "full", None),
            }
        )
        == 1
    )


def test_the_fields_of_an_issue_are_readable_by_name():
    issue = make_issue(cg.UNKNOWN_PROPERTY, "d", source="properties", property="rate")

    assert issue.fields["property"] == "rate"
    assert issue.fields["code"] == issue.code == cg.UNKNOWN_PROPERTY
    assert issue.fields.get("suggested_path") is None
