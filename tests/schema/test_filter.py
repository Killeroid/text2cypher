from text2cypher.schema import GraphSchema, Index, Path, filter_schema

SCHEMA = GraphSchema(
    node_properties={
        "Genre": {"name": ("String",)},
        "Movie": {"title": ("String",), "year": ("Integer",)},
        "Person": {"born": ("Integer",), "name": ("String",)},
        "Studio": {"name": ("String",)},
    },
    relationship_properties={
        "ACTED_IN": {"role": ("String",)},
        "DIRECTED": {},
        "IN_GENRE": {},
        "MADE_BY": {"since": ("Integer",)},
    },
    paths=(
        Path("Movie", "IN_GENRE", "Genre"),
        Path("Movie", "MADE_BY", "Studio"),
        Path("Person", "ACTED_IN", "Movie"),
        Path("Person", "DIRECTED", "Movie"),
    ),
    indexes=(
        Index("Genre", "name", "RANGE"),
        Index("Movie", "title", "FULLTEXT"),
        Index("Person", "name", "RANGE"),
        Index("MADE_BY", "since", "RANGE"),
    ),
)


def names(schema: GraphSchema) -> tuple[set[str], set[str]]:
    return set(schema.node_properties), set(schema.relationship_properties)


def test_a_label_in_the_question_keeps_it_and_what_touches_it():
    schema = filter_schema(SCHEMA, "Which studios exist?")

    assert names(schema) == ({"Movie", "Studio"}, {"MADE_BY"})
    assert schema.paths == (Path("Movie", "MADE_BY", "Studio"),)


def test_the_labels_of_a_kept_path_keep_their_properties():
    schema = filter_schema(SCHEMA, "Which studios exist?")

    assert schema.node_properties["Movie"] == {
        "title": ("String",),
        "year": ("Integer",),
    }
    assert schema.relationship_properties["MADE_BY"] == {"since": ("Integer",)}


def test_a_relationship_type_in_the_question_is_found_by_its_words():
    # "acted" and ACTED_IN share the stem "act"; "in" is not a word to match on
    schema = filter_schema(SCHEMA, "Who acted in Heat?")

    assert names(schema) == ({"Movie", "Person"}, {"ACTED_IN"})
    assert schema.paths == (Path("Person", "ACTED_IN", "Movie"),)


def test_plurals_and_other_endings_match_by_stem():
    assert names(filter_schema(SCHEMA, "How many movies are there?"))[0] >= {"Movie"}
    assert "DIRECTED" in names(filter_schema(SCHEMA, "Who directs films?"))[1]


def test_a_label_written_in_camel_case_is_split_into_words():
    schema = GraphSchema(
        node_properties={"MovieGenre": {"name": ("String",)}, "Person": {}},
        relationship_properties={"LIKES": {}},
        paths=(Path("Person", "LIKES", "MovieGenre"),),
    )

    assert names(filter_schema(schema, "Which genre is liked?"))[0] == {
        "MovieGenre",
        "Person",
    }


def test_the_question_is_matched_without_regard_to_case_or_punctuation():
    schema = filter_schema(SCHEMA, "WHO DIRECTED, EXACTLY?!")

    assert names(schema) == ({"Movie", "Person"}, {"DIRECTED"})


def test_a_question_that_matches_nothing_keeps_the_whole_schema():
    assert filter_schema(SCHEMA, "What is the weather today?") == SCHEMA


def test_words_that_are_not_names_do_not_match_a_name_made_of_such_words():
    # IN_GENRE is made of "in" and "genre"; only "genre" can match
    schema = filter_schema(SCHEMA, "What is in the box?")

    assert schema == SCHEMA


def test_a_property_name_alone_does_not_match():
    # "born" is a property of Person, not a name: nothing matches, nothing is pruned
    assert filter_schema(SCHEMA, "Who was born in 1950?") == SCHEMA


def test_a_label_made_only_of_stopwords_is_still_found_by_its_word():
    schema = GraphSchema({"Show": {"name": ("String",)}, "Venue": {}}, {}, ())

    assert names(filter_schema(schema, "Which shows are on?")) == ({"Show"}, set())


def test_a_relationship_type_made_only_of_stopwords_is_still_found_by_its_word():
    # only the type can match here: no label is named in the question
    schema = GraphSchema(
        {"Person": {}, "Pet": {}, "Club": {}},
        {"HAS": {}, "MEMBER_OF": {}},
        (Path("Person", "HAS", "Pet"), Path("Person", "MEMBER_OF", "Club")),
    )

    assert names(filter_schema(schema, "Who has them?")) == (
        {"Person", "Pet"},
        {"HAS"},
    )


def test_names_with_accents_match_the_question_without_regard_to_them_being_ascii():
    schema = GraphSchema({"Café": {}, "ÉquipeNationale": {}, "Bar": {}}, {}, ())

    assert names(filter_schema(schema, "Which café is open?"))[0] == {"Café"}
    assert names(filter_schema(schema, "Which équipe plays?"))[0] == {"ÉquipeNationale"}


def test_an_acronym_in_a_name_is_a_word_of_its_own():
    schema = GraphSchema({"HTTPServer": {}, "Client": {}}, {}, ())

    assert names(filter_schema(schema, "Which server is it?"))[0] == {"HTTPServer"}


def test_the_indexes_of_what_is_kept_are_kept_and_the_others_go():
    schema = filter_schema(SCHEMA, "Which studios exist?")

    # Studio and Movie are kept, with the type that joins them
    assert set(schema.indexes) == {
        Index("Movie", "title", "FULLTEXT"),
        Index("MADE_BY", "since", "RANGE"),
    }
