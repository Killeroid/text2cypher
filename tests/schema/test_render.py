from dataclasses import replace

from text2cypher.schema import DataDictionary, GraphSchema, Index, Path, render_schema


def test_render_schema():
    schema = GraphSchema(
        node_properties={
            "Genre": {},
            "Movie": {"title": ("String",), "year": ("Integer", "String")},
        },
        relationship_properties={"ACTED_IN": {"role": ("String",)}, "IN_GENRE": {}},
        paths=(Path("Movie", "IN_GENRE", "Genre"), Path("Person", "ACTED_IN", "Movie")),
    )

    assert render_schema(schema) == (
        "Node properties:\n"
        "Genre {}\n"
        "Movie {title: STRING, year: INTEGER|STRING}\n"
        "Relationship properties:\n"
        "ACTED_IN {role: STRING}\n"
        "The relationships:\n"
        "(:Movie)-[:IN_GENRE]->(:Genre)\n"
        "(:Person)-[:ACTED_IN]->(:Movie)"
    )


def test_render_schema_leaves_out_empty_sections():
    schema = GraphSchema(
        node_properties={"Thing": {}},
        relationship_properties={"R": {}},
        paths=(),
    )

    assert render_schema(schema) == "Node properties:\nThing {}"
    assert render_schema(GraphSchema({}, {}, ())) == ""


def test_render_schema_quotes_names_that_need_it():
    schema = GraphSchema(
        node_properties={"My Label": {"a b": ("Integer",)}},
        relationship_properties={},
        paths=(Path("My Label", "rel type", "My Label"),),
    )

    assert render_schema(schema) == (
        "Node properties:\n"
        "`My Label` {`a b`: INTEGER}\n"
        "The relationships:\n"
        "(:`My Label`)-[:`rel type`]->(:`My Label`)"
    )


def test_render_schema_lists_the_indexes_after_the_paths():
    schema = GraphSchema(
        node_properties={"Movie": {"title": ("String",)}},
        relationship_properties={"ACTED_IN": {"role": ("String",)}},
        paths=(Path("Person", "ACTED_IN", "Movie"),),
        indexes=(
            Index("ACTED_IN", "role", "RANGE"),
            Index("Movie", "title", "FULLTEXT"),
            Index("Movie", "title", "RANGE"),
        ),
    )

    assert render_schema(schema).endswith(
        "(:Person)-[:ACTED_IN]->(:Movie)\n"
        "Indexes:\n"
        "ACTED_IN.role: RANGE\n"
        "Movie.title: FULLTEXT, RANGE"
    )


def test_render_schema_quotes_the_names_of_an_index():
    schema = GraphSchema(
        node_properties={"Odd Label": {"my prop": ("String",)}},
        relationship_properties={},
        paths=(),
        indexes=(Index("Odd Label", "my prop", "RANGE"),),
    )

    assert render_schema(schema).endswith("Indexes:\n`Odd Label`.`my prop`: RANGE")


def test_render_schema_has_no_index_section_without_indexes():
    schema = GraphSchema(
        node_properties={"A": {}}, relationship_properties={}, paths=()
    )

    assert "Indexes" not in render_schema(schema)


def test_render_schema_lists_the_indexes_in_order_whatever_order_they_come_in():
    schema = GraphSchema(
        node_properties={"B": {}, "A": {}},
        relationship_properties={},
        paths=(),
        indexes=(
            Index("B", "y", "RANGE"),
            Index("A", "x", "VECTOR"),
            Index("A", "x", "RANGE"),
        ),
    )

    assert render_schema(schema).endswith("Indexes:\nA.x: RANGE, VECTOR\nB.y: RANGE")


def test_render_schema_lists_the_user_defined_functions_last():
    schema = GraphSchema(
        node_properties={"A": {}},
        relationship_properties={},
        paths=(),
        indexes=(Index("A", "x", "RANGE"),),
        functions=("lib.f", "lib.g"),
    )

    assert render_schema(schema).endswith(
        "Indexes:\nA.x: RANGE\nUser-defined functions:\nlib.f\nlib.g"
    )


def test_render_schema_has_no_function_section_without_functions():
    schema = GraphSchema(
        node_properties={"A": {}}, relationship_properties={}, paths=()
    )

    assert "User-defined" not in render_schema(schema)


def test_render_schema_shows_the_examples_after_the_type_of_a_property():
    schema = GraphSchema(
        node_properties={"Movie": {"title": ("String",), "year": ("Integer",)}},
        relationship_properties={"ACTED_IN": {"role": ("String",)}},
        paths=(),
        node_examples={"Movie": {"title": ("'Alien'", "'Heat'")}},
        relationship_examples={"ACTED_IN": {"role": ("'Neo'",)}},
    )

    assert render_schema(schema) == (
        "Node properties:\n"
        "Movie {title: STRING (e.g. 'Alien', 'Heat'), year: INTEGER}\n"
        "Relationship properties:\n"
        "ACTED_IN {role: STRING (e.g. 'Neo')}"
    )


DESCRIBED = GraphSchema(
    node_properties={
        "Movie": {"title": ("String",), "year": ("Integer",)},
        "Person": {},
    },
    relationship_properties={"ACTED_IN": {"role": ("String",)}, "DIRECTED": {}},
    paths=(Path("Person", "ACTED_IN", "Movie"), Path("Person", "DIRECTED", "Movie")),
)


def described(spec: dict) -> GraphSchema:
    return replace(DESCRIBED, data_dictionary=DataDictionary(spec))


def dictionary_of(schema: GraphSchema) -> str:
    """What follows the schema in the prompt: the data dictionary."""
    text = render_schema(schema)
    return text[text.index("Data dictionary:") :] if "Data dictionary:" in text else ""


def test_the_data_dictionary_is_a_section_after_the_schema():
    schema = described(
        {
            "nodes": {
                "Movie": {
                    "description": "A film.",
                    "properties": {"year": "Year of first release."},
                },
                "Person": {"description": "Someone in film."},
            },
            "relationships": {
                "ACTED_IN": {
                    "description": "A cast credit.",
                    "properties": {"role": "Part."},
                }
            },
        }
    )

    assert render_schema(schema) == (
        render_schema(DESCRIBED) + "\n"
        "Data dictionary:\n"
        "Labels:\n"
        "- Movie: A film.\n"
        "  - title\n"
        "  - year: Year of first release.\n"
        "- Person: Someone in film.\n"
        "Relationship types:\n"
        "- ACTED_IN: A cast credit.\n"
        "  - role: Part.\n"
        "- DIRECTED"
    )


def test_everything_the_graph_has_is_listed_once_the_dictionary_says_anything():
    # a description of one label: no other label, type or property looks unimportant
    schema = described({"nodes": {"Person": {"description": "Someone in film."}}})

    assert dictionary_of(schema) == (
        "Data dictionary:\n"
        "Labels:\n"
        "- Movie\n"
        "  - title\n"
        "  - year\n"
        "- Person: Someone in film.\n"
        "Relationship types:\n"
        "- ACTED_IN\n"
        "  - role\n"
        "- DIRECTED"
    )


def test_a_label_with_only_a_property_text_is_listed_with_it():
    schema = described({"nodes": {"Movie": {"properties": {"year": "Release year."}}}})

    assert "- Movie\n  - title\n  - year: Release year.\n- Person\n" in (
        dictionary_of(schema)
    )


def test_a_heading_is_only_there_when_the_graph_has_that_kind_of_name():
    only_a_label = GraphSchema(
        node_properties={"Movie": {}},
        relationship_properties={},
        paths=(),
        data_dictionary=DataDictionary(
            {"nodes": {"Movie": {"description": "A film."}}}
        ),
    )

    assert dictionary_of(only_a_label) == "Data dictionary:\nLabels:\n- Movie: A film."
    assert "Relationship types:" not in render_schema(only_a_label)


def test_a_dictionary_with_no_text_for_the_graph_shows_nothing():
    for spec in (
        {},
        {"nodes": {"Movie": {"description": "", "properties": {"year": " "}}}},
        {"nodes": {"Moive": {"description": "A typo."}}},
        {"nodes": {"Person": {"properties": {"age": "Years."}}}},
    ):
        assert "Data dictionary" not in render_schema(described(spec))


def test_a_schema_without_a_data_dictionary_renders_as_it_always_did():
    assert render_schema(DESCRIBED) == render_schema(
        replace(DESCRIBED, data_dictionary=None)
    )
    assert "Data dictionary" not in render_schema(DESCRIBED)
    assert "Data dictionary" not in render_schema(described({}))


def test_a_name_the_graph_lacks_is_left_out_and_the_ones_it_has_are_not():
    schema = described(
        {
            "nodes": {
                "Moive": {"description": "A typo."},
                "Movie": {"properties": {"year": "Year.", "yaer": "A typo."}},
            },
            "relationships": {"ACTED": {"description": "A typo."}},
        }
    )

    text = dictionary_of(schema)
    assert "Moive" not in text
    assert "yaer" not in text
    assert "ACTED:" not in text
    assert "- Movie\n  - title\n  - year: Year.\n" in text
    assert "- ACTED_IN\n  - role\n- DIRECTED" in text


def test_a_relationship_type_known_only_from_its_paths_can_be_described():
    schema = replace(
        DESCRIBED,
        relationship_properties={},
        data_dictionary=DataDictionary(
            {"relationships": {"DIRECTED": {"description": "Led."}}}
        ),
    )

    assert dictionary_of(schema).endswith(
        "Relationship types:\n- ACTED_IN\n- DIRECTED: Led."
    )


def test_a_label_and_a_relationship_type_of_one_name_are_shown_for_what_each_is():
    schema = GraphSchema(
        node_properties={"HAS": {}},
        relationship_properties={"HAS": {}},
        paths=(Path("HAS", "HAS", "HAS"),),
        data_dictionary=DataDictionary(
            {
                "nodes": {"HAS": {"description": "A label."}},
                "relationships": {"HAS": {"description": "A type."}},
            }
        ),
    )

    assert dictionary_of(schema) == (
        "Data dictionary:\nLabels:\n- HAS: A label.\nRelationship types:\n- HAS: A type."
    )


def test_the_names_of_the_data_dictionary_are_quoted_as_the_rest_of_the_schema_is():
    schema = GraphSchema(
        node_properties={"My Label": {"a b": ("Integer",), "c": ("Integer",)}},
        relationship_properties={},
        paths=(),
        data_dictionary=DataDictionary(
            {
                "nodes": {
                    "My Label": {"description": "Odd.", "properties": {"a b": "Count."}}
                }
            }
        ),
    )

    assert dictionary_of(schema) == (
        "Data dictionary:\nLabels:\n- `My Label`: Odd.\n  - `a b`: Count.\n  - c"
    )


def test_without_the_properties_only_labels_and_relationship_types_are_listed():
    from text2cypher.schema._render import as_text, data_dictionary_section

    schema = described(
        {
            "nodes": {
                "Movie": {"description": "A film.", "properties": {"year": "Y."}}
            },
        }
    )

    section = data_dictionary_section(schema, properties=False)

    assert section is not None
    assert as_text(section) == (
        "Data dictionary:\n"
        "Labels:\n- Movie: A film.\n- Person\n"
        "Relationship types:\n- ACTED_IN\n- DIRECTED"
    )
    # property texts alone are not text for it
    only_properties = described({"nodes": {"Movie": {"properties": {"year": "Y."}}}})
    assert data_dictionary_section(only_properties, properties=False) is None
