"""Fix a misspelled name in a query without asking the LLM."""

import re
from collections.abc import Iterable, Mapping

from cypher_guardrail.issues import UNKNOWN_LABEL, UNKNOWN_PROPERTY, UNKNOWN_REL_TYPE

from text2cypher._generation.problems import likely_typos
from text2cypher._validation import Issue
from text2cypher.schema import GraphSchema
from text2cypher.schema._model import quote_identifier

# What is not code: a string, a line comment, a block comment. A name inside one is left alone.
_LITERALS = re.compile(
    r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|//[^\n]*|/\*.*?\*/)""", re.DOTALL
)


def fix_typos(query: str, issues: Iterable[Issue], schema: GraphSchema) -> str | None:
    """``query`` with each misspelled name replaced by the one name it is a typo of.

    A label, relationship type or property that the validators report as unknown exists
    nowhere in the graph, so every use of it is wrong and each is replaced, wherever it is
    in the query's code. A name is fixed only when exactly one name of the schema is a
    likely typo of it (the same rule as the "did you mean" in the problems); an ambiguous
    one is left. A property that exists on another label is not touched: which variable
    it belongs to is not known. Returns ``None`` when nothing was changed.
    """
    properties = {
        prop
        for props in (
            *schema.node_properties.values(),
            *schema.relationship_properties.values(),
        )
        for prop in props
    }
    names = {
        UNKNOWN_LABEL: ("label", schema.node_properties),
        UNKNOWN_REL_TYPE: ("relationship_type", schema.relationship_properties),
        UNKNOWN_PROPERTY: ("property", properties),
    }
    renames: dict[str, dict[str, str]] = {code: {} for code in names}
    for issue in issues:
        if issue.code in names:
            field, known = names[issue.code]
            wrong = issue.fields[field]
            if (right := _the_one(wrong, known)) is not None:
                renames[issue.code][wrong] = right
    fixed = _rewrite(query, renames)
    return None if fixed == query else fixed


def _the_one(name: str, known: Iterable[str]) -> str | None:
    likely = likely_typos(name, known)
    return likely[0][1] if len(likely) == 1 else None


def _rewrite(query: str, renames: Mapping[str, Mapping[str, str]]) -> str:
    parts = _LITERALS.split(query)
    # An odd part is a literal (the split keeps what it matched), an even one is code
    parts[::2] = [_rewrite_code(code, renames) for code in parts[::2]]
    return "".join(parts)


def _rewrite_code(code: str, renames: Mapping[str, Mapping[str, str]]) -> str:
    # A label or a relationship type follows a colon (or a bar: [:A|B]); a property follows a
    # dot, or is a key before a colon in a map. Backticks around the old name go with it.
    for wrong, right in (
        *renames[UNKNOWN_LABEL].items(),
        *renames[UNKNOWN_REL_TYPE].items(),
    ):
        code = _replace(code, r"([:|]\s*:?\s*)", wrong, right, r"")
    for wrong, right in renames[UNKNOWN_PROPERTY].items():
        code = _replace(code, r"(\.\s*)", wrong, right, r"(?!\s*\()")
        code = _replace(code, r"([{,]\s*)", wrong, right, r"(?=\s*:)")
    return code


def _replace(code: str, before: str, wrong: str, right: str, after: str) -> str:
    pattern = re.compile(rf"{before}(`?){re.escape(wrong)}\2(?![\w`]){after}")
    return pattern.sub(lambda found: found[1] + quote_identifier(right), code)
