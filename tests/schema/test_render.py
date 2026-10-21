from text2cypher.schema import GraphSchema, Index, Path, render_schema


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
