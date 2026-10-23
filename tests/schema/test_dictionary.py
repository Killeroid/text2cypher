import pytest

from text2cypher.schema import DataDictionary, GraphSchema, Path

SCHEMA = GraphSchema(
    node_properties={
        "Movie": {"title": ("String",), "year": ("Integer",)},
        "Person": {"born": ("Integer",), "name": ("String",)},
    },
    relationship_properties={"ACTED_IN": {"role": ("String",)}, "DIRECTED": {}},
    paths=(Path("Person", "ACTED_IN", "Movie"), Path("Person", "DIRECTED", "Movie")),
)

FULL = {
    "nodes": {
        "Movie": {"description": "A film.", "properties": {"year": "Release year."}},
        "Person": {"description": "Someone in film."},
    },
    "relationships": {
        "ACTED_IN": {"description": "A cast credit.", "properties": {"role": "Part."}}
    },
}


def test_a_dictionary_holds_what_was_said_of_each_label_and_relationship_type():
    dictionary = DataDictionary(FULL)

    assert dictionary.nodes["Movie"].description == "A film."
    assert dictionary.nodes["Movie"].properties == {"year": "Release year."}
    assert dictionary.relationships["ACTED_IN"].properties == {"role": "Part."}


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"nodes": {}},
        {"relationships": {"ACTED_IN": {"description": "A cast credit."}}},
        {"nodes": {"Movie": {}}},
        {"nodes": {"Movie": {"properties": {"year": "Release year."}}}},
        {"nodes": {"Movie": {"description": "A film."}}},
    ],
)
def test_everything_is_optional(spec):
    DataDictionary(spec)


def test_a_description_that_was_not_given_is_not_blank():
    dictionary = DataDictionary({"nodes": {"Movie": {"properties": {"year": "Year."}}}})

    assert dictionary.nodes["Movie"].description is None
    assert dictionary.check(SCHEMA) == []


def test_whitespace_in_a_text_is_collapsed():
    dictionary = DataDictionary(
        {"nodes": {"Movie": {"description": "  A film,\n  not a\tshow.  "}}}
    )

    assert dictionary.nodes["Movie"].description == "A film, not a show."


def test_it_does_not_change_when_the_dict_it_was_built_from_does():
    spec = {"nodes": {"Movie": {"properties": {"year": "Year."}}}}
    dictionary = DataDictionary(spec)

    spec["nodes"]["Movie"]["properties"]["year"] = "changed"

    assert dictionary.nodes["Movie"].properties == {"year": "Year."}


@pytest.mark.parametrize(
    "spec",
    [
        {"nodes": {"Movie": {"description": 7}}},
        {"nodes": {"Movie": {"description": None}}},
        {"nodes": {"Movie": {"properties": {"year": 1999}}}},
        {"relationships": {"ACTED_IN": {"description": ["A cast credit."]}}},
        {"nodes": {1: {"description": "A film."}}},
        {"nodes": {"Movie": {"properties": {1: "Year."}}}},
        {"nodes": ["Movie"]},
        {"nodes": {"Movie": "A film."}},
        {"nodes": {"Movie": {"properties": ["year"]}}},
    ],
)
def test_a_value_of_the_wrong_type_is_an_error(spec):
    with pytest.raises(TypeError):
        DataDictionary(spec)


def test_the_dictionary_itself_must_be_a_mapping():
    with pytest.raises(TypeError):
        DataDictionary([("nodes", {})])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("spec", "unknown"),
    [
        ({"edges": {}}, "edges"),
        ({"nodes": {"Movie": {"descripton": "A film."}}}, "descripton"),
        ({"nodes": {"Movie": {"props": {}}}}, "props"),
    ],
)
def test_a_key_that_is_not_one_of_its_own_is_an_error_that_names_it(spec, unknown):
    with pytest.raises(ValueError, match=unknown):
        DataDictionary(spec)


def test_a_blank_description_is_accepted():
    dictionary = DataDictionary(
        {"nodes": {"Movie": {"description": "  ", "properties": {"year": ""}}}}
    )

    assert dictionary.nodes["Movie"].description == ""
    assert dictionary.nodes["Movie"].properties == {"year": ""}


def test_check_has_nothing_to_say_about_a_dictionary_that_fits_the_graph():
    assert DataDictionary(FULL).check(SCHEMA) == []


def test_check_reports_blank_descriptions():
    dictionary = DataDictionary(
        {
            "nodes": {"Movie": {"description": "", "properties": {"year": " "}}},
            "relationships": {"DIRECTED": {"description": "\n"}},
        }
    )

    assert dictionary.check(SCHEMA) == [
        "node 'Movie' has a blank description",
        "node 'Movie': property 'year' has a blank description",
        "relationship 'DIRECTED' has a blank description",
    ]


def test_check_reports_names_the_graph_does_not_have():
    dictionary = DataDictionary(
        {
            "nodes": {
                "Moive": {"description": "A film."},
                "Person": {"properties": {"age": "Years.", "born": "Year."}},
            },
            "relationships": {
                "ACTED": {"description": "A cast credit."},
                "ACTED_IN": {"properties": {"pay": "Fee."}},
            },
        }
    )

    assert dictionary.check(SCHEMA) == [
        "node 'Moive' is not in the graph",
        "node 'Person' has no property 'age'",
        "relationship 'ACTED' is not in the graph",
        "relationship 'ACTED_IN' has no property 'pay'",
    ]


def test_a_relationship_type_that_has_a_path_but_no_properties_is_in_the_graph():
    dictionary = DataDictionary(
        {"relationships": {"DIRECTED": {"description": "Led."}}}
    )

    assert dictionary.check(SCHEMA) == []


def test_a_label_and_a_relationship_type_of_one_name_are_checked_apart():
    dictionary = DataDictionary({"relationships": {"Movie": {"description": "?"}}})

    assert dictionary.check(SCHEMA) == ["relationship 'Movie' is not in the graph"]


def test_only_keeps_the_entities_asked_for_and_what_was_said_of_them():
    dictionary = DataDictionary(FULL)

    kept = dictionary.only({"Movie"}, set())

    assert kept == DataDictionary(
        {"nodes": {"Movie": FULL["nodes"]["Movie"]}, "relationships": {}}
    )


def test_only_keeps_a_description_that_was_not_given_as_not_given_and_a_blank_as_blank():
    dictionary = DataDictionary(
        {
            "nodes": {
                "Movie": {"properties": {"year": "Year."}},
                "Person": {"description": ""},
            }
        }
    )

    kept = dictionary.only({"Movie", "Person"}, set())

    assert kept.nodes["Movie"].description is None
    assert kept.nodes["Person"].description == ""
    assert kept == dictionary
