import pytest

from text2cypher.schema._model import Path, quote_identifier


@pytest.mark.parametrize(
    ("name", "quoted"),
    [
        ("Person", "Person"),
        ("_x1", "_x1"),
        ("My Label", "`My Label`"),
        ("1abc", "`1abc`"),
        ("naïve", "`naïve`"),
        ("", "``"),
    ],
)
def test_quote_identifier(name: str, quoted: str):
    assert quote_identifier(name) == quoted


def test_quote_identifier_rejects_a_backtick():
    with pytest.raises(ValueError, match="backtick"):
        quote_identifier("a`b")


def test_a_path_writes_itself_as_a_pattern_with_quoted_names():
    assert (
        Path("Person", "ACTED_IN", "Movie").pattern()
        == "(:Person)-[:ACTED_IN]->(:Movie)"
    )
    assert (
        Path("Odd Label", "HAS A SPACE", "B").pattern()
        == "(:`Odd Label`)-[:`HAS A SPACE`]->(:B)"
    )
