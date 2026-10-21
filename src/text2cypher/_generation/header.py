"""Repair the ``CYPHER`` header of a query that uses parameters."""

import re

_KEYWORD = re.compile(r"cypher(?!\w)", re.IGNORECASE)
_PAIR = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*")
_WORD = re.compile(r"[^\s,]+")


def repair_header(query: str) -> str | None:
    """``query`` with its parameter header in the form FalkorDB accepts, or ``None``.

    FalkorDB wants ``CYPHER`` and then the pairs separated by spaces, not commas, so
    ``name='Ann', year=1990 MATCH ...`` is a syntax error, and so is a header without
    ``CYPHER``. This adds a missing ``CYPHER`` and writes the pairs with one space between.
    A query with no header, or one whose values cannot be read to their end, gives ``None``,
    and so does one that is already written like that.
    """
    text = query.strip()
    keyword = _KEYWORD.match(text)
    rest = text[keyword.end() :] if keyword else text
    pairs: list[str] = []
    position = 0
    while found := _PAIR.match(rest, position):
        end = _value_end(rest, found.end())
        if end is None:
            return None
        pairs.append(f"{found[1]}={rest[found.end() : end]}")
        position = end
        while position < len(rest) and (
            rest[position].isspace() or rest[position] == ","
        ):
            position += 1
    if not pairs:
        return None
    repaired = f"CYPHER {' '.join(pairs)} {rest[position:]}".rstrip()
    return None if repaired == text else repaired


def _value_end(text: str, start: int) -> int | None:
    """Where the value that begins at ``start`` ends: a string, a list, a map or a word."""
    if start >= len(text):
        return None
    if text[start] in "'\"[{(":
        return _closing(text, start)
    word = _WORD.match(text, start)
    return word.end() if word else None


def _closing(text: str, start: int) -> int | None:
    """The end of the string or bracketed value that begins at ``start``, or ``None``."""
    depth, quote, i = 0, None, start
    while i < len(text):
        char = text[i]
        if quote and char == "\\":
            i += 2
            continue
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
        i += 1
        if quote is None and depth == 0:
            return i
    return None
