import pytest

from text2cypher._generation import extract_cypher


@pytest.mark.parametrize(
    ("reply", "query"),
    [
        ("```cypher\nMATCH (n) RETURN n\n```", "MATCH (n) RETURN n"),
        ("```Cypher\nMATCH (n) RETURN n\n```", "MATCH (n) RETURN n"),
        ("```\nMATCH (n) RETURN n\n```", "MATCH (n) RETURN n"),
        ("```MATCH (n) RETURN n```", "MATCH (n) RETURN n"),
        # Windows line endings
        ("```sql\r\nMATCH (n) RETURN n\r\n```", "MATCH (n) RETURN n"),
        ("```cypher\r\nMATCH (n)\r\nRETURN n\r\n```", "MATCH (n)\r\nRETURN n"),
        # a language tag on a one-line block is a tag, not part of the query
        ("```cypher MATCH (n) RETURN n```", "MATCH (n) RETURN n"),
        ("```CYPHER MATCH (n) RETURN n```", "MATCH (n) RETURN n"),
        ("```opencypher MATCH (n) RETURN n```", "MATCH (n) RETURN n"),
        ("```cypher```", ""),
        # ...but only a known tag followed by a space, not the start of a word
        ("```cypherMATCH (n) RETURN n```", "cypherMATCH (n) RETURN n"),
        (
            "Here is the query:\n```cypher\nMATCH (n)\nRETURN n\n```\nHope it helps!",
            "MATCH (n)\nRETURN n",
        ),
        # only the first block is the answer
        (
            "```cypher\nMATCH (a) RETURN a\n```\nor\n```cypher\nMATCH (b) RETURN b\n```",
            "MATCH (a) RETURN a",
        ),
        ("  MATCH (n) RETURN n \n", "MATCH (n) RETURN n"),
        # no unanswerable marker is needed: an empty block is an empty query
        ("```cypher\n```", ""),
        ("```\n```", ""),
        ("", ""),
        # a reply cut off before the closing fence is left as it is
        ("```cypher\nMATCH (n) RETURN n", "```cypher\nMATCH (n) RETURN n"),
    ],
)
def test_extract_cypher(reply: str, query: str):
    assert extract_cypher(reply) == query
