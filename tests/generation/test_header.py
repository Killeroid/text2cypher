import pytest

from text2cypher._generation import repair_header

BODY = "MATCH (p:Person {name: $name}) RETURN p.name"


@pytest.mark.parametrize(
    ("query", "repaired"),
    [
        # commas between the pairs
        (
            f"CYPHER name='Ann', year=1990 {BODY}",
            f"CYPHER name='Ann' year=1990 {BODY}",
        ),
        # a missing CYPHER
        (f"name='Ann' {BODY}", f"CYPHER name='Ann' {BODY}"),
        # both
        (
            f"name='Ann' , year=1990 {BODY}",
            f"CYPHER name='Ann' year=1990 {BODY}",
        ),
        # a comma inside a string or a list is a part of the value
        (
            f"CYPHER ids=[1,2,3], note='a, b' {BODY}",
            f"CYPHER ids=[1,2,3] note='a, b' {BODY}",
        ),
        (
            f'CYPHER note="it\'s, ok", n=2 {BODY}',
            f'CYPHER note="it\'s, ok" n=2 {BODY}',
        ),
        (
            f"CYPHER m={{a: 1, b: 'x'}}, n=2 {BODY}",
            f"CYPHER m={{a: 1, b: 'x'}} n=2 {BODY}",
        ),
        # a quote that is escaped does not end the string
        (
            f"CYPHER note='it\\'s', n=2 {BODY}",
            f"CYPHER note='it\\'s' n=2 {BODY}",
        ),
        # numbers, booleans and null are words
        (
            f"CYPHER n=-1.5, ok=true, nothing=null {BODY}",
            f"CYPHER n=-1.5 ok=true nothing=null {BODY}",
        ),
        # the query itself is not touched, and a newline in it stays
        (
            "name='Ann',\nCYPHER_X=1\nMATCH (n)\nRETURN n",
            "CYPHER name='Ann' CYPHER_X=1 MATCH (n)\nRETURN n",
        ),
    ],
)
def test_the_header_is_written_the_way_falkordb_accepts(query: str, repaired: str):
    assert repair_header(query) == repaired


@pytest.mark.parametrize(
    "query",
    [
        f"CYPHER name='Ann' {BODY}",
        f"CYPHER name='Ann' year=1990 {BODY}",
        BODY,
        "RETURN x = 1",
        "WHERE a = 1 RETURN a",
        "",
        "CYPHER name='Ann'",
        # a value that is never closed cannot be read, so nothing is guessed
        f"CYPHER name='Ann, year=1990 {BODY}",
        f"CYPHER ids=[1, 2 {BODY}",
    ],
)
def test_a_query_that_has_nothing_to_repair_is_left_alone(query: str):
    assert repair_header(query) is None


def test_repairing_twice_changes_nothing_more():
    once = repair_header(f"name='Ann', year=1990 {BODY}")

    assert once is not None
    assert repair_header(once) is None
