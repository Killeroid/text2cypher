"""Put what is wrong with a query to the LLM, in words it can act on.

cypher-guardrail's descriptions are written for developers (some are FalkorDB's own error
text). This words each issue from its structured fields, and uses the schema the query was
written for to say where a missing name can be found, or what it may have been meant to be.
Issues without special wording keep their description.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any

from cypher_guardrail.issues import (
    AGGREGATE_TYPE_MISMATCH,
    CARTESIAN_PRODUCT,
    INVALID_LIMIT_ERROR,
    INVALID_PROPERTY_ACCESS,
    PARAMETER_NOT_PROVIDED,
    PROPERTY_TYPE_MISMATCH,
    UNBOUND_VARIABLE_ERROR,
    UNBOUNDED_PATH,
    UNKNOWN_FUNCTION,
    UNKNOWN_FUNCTION_ERROR,
    UNKNOWN_LABEL,
    UNKNOWN_PATH,
    UNKNOWN_PROPERTY,
    UNKNOWN_REL_TYPE,
    UNLABELED_NODE_SCAN,
    WRITE_QUERY_ERROR,
)

from text2cypher._validation import Issue
from text2cypher.schema import GraphSchema

# The relationship type of a one-hop pattern as cypher-guardrail writes it: (:A)-[:T]->(:B)
_PATTERN_TYPE = re.compile(r"\[:(`[^`]*`|[^\]`]+)\]")
_MOST_PATHS = 6
_SURE, _LIKELY, _PREFIX = 0.75, 0.6, 3

type Fields = Mapping[str, Any]


def problems(issues: Iterable[Issue], schema: GraphSchema) -> str:
    """One bullet per distinct problem, with its hint lines, for the corrector prompt."""
    issues = list(issues)
    # "No label has this property" says less than "this label has no such property"
    named = {
        prop
        for issue in issues
        if issue.code == INVALID_PROPERTY_ACCESS
        for prop in issue.fields["properties"]
    }
    return "\n".join(
        dict.fromkeys(
            _describe(issue, schema)
            for issue in issues
            if not (
                issue.code == UNKNOWN_PROPERTY and issue.fields["property"] in named
            )
        )
    )


def _describe(issue: Issue, schema: GraphSchema) -> str:
    handler = _HANDLERS.get(issue.code)
    if handler and (text := handler(issue.fields, schema)):
        return text
    return _plain(issue.description, issue.fields.get("suggested_path"))


def _plain(text: str, suggested: str | None) -> str:
    description = f"- {text}"
    if suggested and suggested not in text:
        description += f"\n  Suggested: {suggested}"
    return description


def _closest(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate that ``name`` is most likely a misspelling of, if there is one.

    A likely typo is very similar, or fairly similar and starts the same way (``rate`` for
    ``rating``); sharing a few letters is not enough (``Client`` and ``City``).
    """
    best, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
        starts_alike = name.lower()[:_PREFIX] == candidate.lower()[:_PREFIX]
        if ratio > best_ratio and (
            ratio >= _SURE or (ratio >= _LIKELY and starts_alike)
        ):
            best, best_ratio = candidate, ratio
    return best


def _unknown(what: str, name: str, known: Mapping[str, Any]) -> str:
    text = f"- There is no {what} `{name}` in the schema."
    if close := _closest(name, known):
        text += f"\n  Did you mean `{close}`?"
    return text


def _unknown_label(raw: Fields, schema: GraphSchema) -> str:
    return _unknown("label", raw["label"], schema.node_properties)


def _unknown_relationship_type(raw: Fields, schema: GraphSchema) -> str:
    return _unknown(
        "relationship type", raw["relationship_type"], schema.relationship_properties
    )


def _parameter_without_value(raw: Fields, schema: GraphSchema) -> str:
    names = [f"`${name}`" for name in raw["parameters"]]
    many = len(names) > 1
    return (
        f"- The query uses the {'parameters' if many else 'parameter'} "
        f"{', '.join(names)}, which {'have' if many else 'has'} no value. "
        "Write the value in the query instead."
    )


def _unbound_variable(raw: Fields, schema: GraphSchema) -> str:
    return (
        f"- The variable `{raw['variable']}` is used but never defined. Define it in a "
        "MATCH or a WITH before it is used, or use one that is defined."
    )


def _unknown_function(raw: Fields, schema: GraphSchema) -> str:
    return f"- There is no function `{raw['name']}` in this FalkorDB. Use only functions that exist."


def _invalid_limit(raw: Fields, schema: GraphSchema) -> str:
    return f"- The value after {raw['clause']} must be a positive integer, written in the query."


def _cartesian_product(raw: Fields, schema: GraphSchema) -> str:
    groups = " and ".join(f"with ({', '.join(group)})" for group in raw["variables"])
    return (
        f"- The patterns {groups} are not connected, so every row of one is paired with "
        "every row of the other. Connect them with a relationship in one pattern."
    )


def _unlabeled_node_scan(raw: Fields, schema: GraphSchema) -> str:
    return (
        f"- `{raw['variable']}` has no label, so every node in the graph is read. "
        "Give it the label it should have."
    )


def _unbounded_path(raw: Fields, schema: GraphSchema) -> str:
    return (
        f"- The path {raw['path']} has no upper bound on its length, so it can read the "
        "whole graph. Give it one, such as *1..4."
    )


def _property_type_mismatch(raw: Fields, schema: GraphSchema) -> str:
    return (
        f"- `{raw['name']}.{raw['property']}` holds {', '.join(raw['expected_types'])} "
        f"values, but the query compares it with the {raw['value_type']} {raw['value']!r}. "
        "Compare it with a value of the right type."
    )


def _aggregate_type_mismatch(raw: Fields, schema: GraphSchema) -> str:
    return (
        f"- `{raw['function']}` needs numbers, but `{raw['name']}.{raw['property']}` holds "
        f"only {', '.join(raw['types'])} values. Use a property that holds numbers."
    )


def _write_query(raw: Fields, schema: GraphSchema) -> str:
    return "- The query can modify the graph. Only read queries are allowed."


def _owners(prop: str, schema: GraphSchema) -> list[str]:
    """Where ``prop`` is a property: the labels and relationship types that have it."""
    labels = [f"`{n}`" for n, p in schema.node_properties.items() if prop in p]
    types = [f"`{n}`" for n, p in schema.relationship_properties.items() if prop in p]
    found = []
    if labels:
        found.append(
            f"the {'label' if len(labels) == 1 else 'labels'} {', '.join(labels)}"
        )
    if types:
        found.append(
            f"the {'relationship type' if len(types) == 1 else 'relationship types'} "
            f"{', '.join(types)}"
        )
    return found


def _unknown_property(raw: Fields, schema: GraphSchema) -> str:
    prop = raw["property"]
    text = f"- No label or relationship type has a property `{prop}`."
    every = {
        name: owner
        for owner, props in (
            *schema.node_properties.items(),
            *schema.relationship_properties.items(),
        )
        for name in props
    }
    if close := _closest(prop, every):
        text += f"\n  Did you mean `{close}` (a property of `{every[close]}`)?"
    return text


def _invalid_property_access(raw: Fields, schema: GraphSchema) -> str:
    kind, name = raw["kind"], raw["name"]
    own = schema.node_properties if kind == "node" else schema.relationship_properties
    what = "label" if kind == "node" else "relationship type"
    lines = []
    for prop in raw["properties"]:
        lines.append(f"- The {what} `{name}` has no property `{prop}`.")
        if owners := _owners(prop, schema):
            lines.append(f"  `{prop}` is a property of {' and '.join(owners)}.")
        if close := _closest(prop, own.get(name, ())):
            lines.append(f"  Did you mean `{close}`?")
        elif own.get(name):
            lines.append(f"  `{name}` has: {', '.join(own[name])}.")
    return "\n".join(lines)


def _unknown_path(raw: Fields, schema: GraphSchema) -> str | None:
    """The problem with a pattern that does not exist, or ``None`` if it has no type."""
    path = raw["path"]
    found = _PATTERN_TYPE.search(path)
    if not found:
        return None
    kind = found.group(1).strip("`")
    where = [p for p in schema.paths if p.type == kind]
    text = f"- The pattern {path} does not exist in the schema."
    if where:
        shown = ", ".join(p.pattern() for p in where[:_MOST_PATHS])
        text += f"\n  `{kind}` goes between: {shown}"
        if len(where) > _MOST_PATHS:
            text += f" and {len(where) - _MOST_PATHS} more"
    return text


# What to say for each code; a handler may return None to fall back to the description
_HANDLERS: dict[str, Callable[[Fields, GraphSchema], str | None]] = {
    UNKNOWN_LABEL: _unknown_label,
    UNKNOWN_REL_TYPE: _unknown_relationship_type,
    UNKNOWN_PROPERTY: _unknown_property,
    INVALID_PROPERTY_ACCESS: _invalid_property_access,
    UNKNOWN_PATH: _unknown_path,
    PARAMETER_NOT_PROVIDED: _parameter_without_value,
    WRITE_QUERY_ERROR: _write_query,
    UNBOUND_VARIABLE_ERROR: _unbound_variable,
    UNKNOWN_FUNCTION_ERROR: _unknown_function,
    UNKNOWN_FUNCTION: _unknown_function,
    INVALID_LIMIT_ERROR: _invalid_limit,
    CARTESIAN_PRODUCT: _cartesian_product,
    UNLABELED_NODE_SCAN: _unlabeled_node_scan,
    UNBOUNDED_PATH: _unbounded_path,
    PROPERTY_TYPE_MISMATCH: _property_type_mismatch,
    AGGREGATE_TYPE_MISMATCH: _aggregate_type_mismatch,
}
