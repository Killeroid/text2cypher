"""Keep what a question needs of a schema, by matching its words to names."""

import re
from collections.abc import Callable, Iterable, Mapping

import snowballstemmer

from text2cypher.schema._model import GraphSchema
from text2cypher.schema._stopwords import STOPWORDS

# Keeps what a question needs of the full schema
type SchemaFilter = Callable[[GraphSchema, str], GraphSchema]

# A word is letters and digits of any script; underscores, spaces and other symbols separate words
_TOKENS = re.compile(r"[^\W_]+")
_stem = snowballstemmer.stemmer("english").stemWord


def filter_schema(schema: GraphSchema, question: str) -> GraphSchema:
    """The part of ``schema`` that ``question`` is about.

    A label or relationship type is in the question when one of its words has the
    same stem as one of the question's words (``acted`` finds ``ACTED_IN``, ``movies``
    finds ``Movie``). A match keeps the paths it is part of, and so the labels and
    relationship types on those paths, with their properties. Property names are not
    matched. A question that matches nothing keeps the whole schema.
    """
    wanted = {_stem(word) for word in _TOKENS.findall(question.lower())}
    labels = _matching(schema.node_properties, wanted)
    types = _matching(schema.relationship_properties, wanted)
    if not labels and not types:
        return schema
    return _restrict(schema, labels, types)


def _matching(names: Iterable[str], wanted: set[str]) -> set[str]:
    return {name for name in names if _stems(name) & wanted}


def _restrict(schema: GraphSchema, labels: set[str], types: set[str]) -> GraphSchema:
    """The paths that touch a matched label or type, and everything those paths use."""
    paths = tuple(
        path
        for path in schema.paths
        if path.source in labels or path.target in labels or path.type in types
    )
    kept_labels = labels | {end for path in paths for end in (path.source, path.target)}
    kept_types = types | {path.type for path in paths}
    kept = kept_labels | kept_types
    return GraphSchema(
        node_properties=_only(schema.node_properties, kept_labels),
        relationship_properties=_only(schema.relationship_properties, kept_types),
        paths=paths,
        indexes=tuple(index for index in schema.indexes if index.entity in kept),
    )


def _only[V](entities: Mapping[str, V], keep: set[str]) -> dict[str, V]:
    return {name: value for name, value in entities.items() if name in keep}


def _words(name: str) -> list[str]:
    """The words of a name: split at underscores, spaces and symbols, at camelCase and
    PascalCase boundaries (``MovieGenre``), after an acronym (``HTTPServer``) and
    between letters and digits."""
    words: list[str] = []
    for token in _TOKENS.findall(name):
        start = 0
        for i in range(1, len(token)):
            if _starts_a_word(token[i - 1], token[i], token[i + 1 : i + 2]):
                words.append(token[start:i])
                start = i
        words.append(token[start:])
    return words


def _starts_a_word(before: str, char: str, after: str) -> bool:
    """Whether a word of a name begins at ``char``, given the characters around it."""
    return (
        before.isdigit() != char.isdigit()
        or ((before.islower() or before.isdigit()) and char.isupper())
        or (before.isupper() and char.isupper() and after.islower())
    )


def _stems(name: str) -> set[str]:
    """The stems of the words of a name, without the stopwords unless it is only those."""
    words = [word.lower() for word in _words(name)]
    return {_stem(word) for word in [w for w in words if w not in STOPWORDS] or words}
