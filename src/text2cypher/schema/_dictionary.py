"""What the user says each part of a graph means, for the LLM to read with the schema."""

from collections.abc import Container, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from text2cypher.schema._model import GraphSchema

_SECTIONS = ("nodes", "relationships")
_ENTRY_KEYS = ("description", "properties")


class Entry(NamedTuple):
    """What is said of one label or relationship type.

    Attributes:
        description: What it is: ``None`` if none was given, ``""`` if it was given blank
            (kept so ``DataDictionary.check`` can report it).
        properties: What each of its properties holds, by name.
    """

    description: str | None
    properties: Mapping[str, str]


@dataclass(frozen=True, slots=True, init=False)
class DataDictionary:
    """Descriptions of the labels, relationship types and properties of a graph.

    It is built from a nested dict in which everything is optional: a whole section, an
    entity, an entity's description or its properties, a single property. Whatever is not
    described is shown to the LLM as the schema shows it, by name alone.

    >>> DataDictionary({
    ...     "nodes": {"Movie": {"description": "A film.", "properties": {"year": "Release year."}}},
    ...     "relationships": {"ACTED_IN": {"description": "A cast credit."}},
    ... })  # doctest: +SKIP

    Whitespace in a text is collapsed to single spaces. A blank text is accepted and shown
    as no text; ``check`` reports it. The descriptions go to the LLM provider, like
    the schema does: use only text you trust.

    Attributes:
        nodes: What is said of each label.
        relationships: What is said of each relationship type (not of each path).

    Raises:
        TypeError: If a description or a name is not a string, or a section, an entity or
            its properties is not a mapping.
        ValueError: If a key is not one of the dictionary's: ``nodes`` and
            ``relationships``, then ``description`` and ``properties``.
    """

    nodes: Mapping[str, Entry]
    relationships: Mapping[str, Entry]

    def __init__(self, spec: Mapping[str, Any]) -> None:
        spec = _mapping(spec, "the data dictionary")
        _only_keys(spec, _SECTIONS, "the data dictionary")
        object.__setattr__(self, "nodes", _entries(spec.get("nodes", {}), "nodes"))
        object.__setattr__(
            self,
            "relationships",
            _entries(spec.get("relationships", {}), "relationships"),
        )

    def only(self, labels: Container[str], types: Container[str]) -> "DataDictionary":
        """The dictionary of the given labels and relationship types alone."""
        return DataDictionary(
            {
                "nodes": _spec(self.nodes, labels),
                "relationships": _spec(self.relationships, types),
            }
        )

    def check(self, schema: "GraphSchema") -> list[str]:
        """What is wrong with the dictionary for ``schema``, one message each.

        That is a label, relationship type or property that ``schema`` does not have (a typo,
        or the graph has changed), which the prompt leaves out, and a description that is
        blank, which it shows as no description. Nothing is raised: the run goes on. Call it
        to catch the mistakes, for example in a test.
        """
        types = {*schema.relationship_properties, *(path.type for path in schema.paths)}
        return [
            *_problems(
                "node", self.nodes, schema.node_properties, schema.node_properties
            ),
            *_problems(
                "relationship",
                self.relationships,
                types,
                schema.relationship_properties,
            ),
        ]


def _problems(
    kind: str,
    entries: Mapping[str, Entry],
    known: Container[str],
    properties: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> list[str]:
    messages = []
    for name, entry in entries.items():
        if name not in known:
            messages.append(f"{kind} {name!r} is not in the graph")
            continue
        if entry.description == "":
            messages.append(f"{kind} {name!r} has a blank description")
        for prop, text in entry.properties.items():
            if prop not in properties.get(name, {}):
                messages.append(f"{kind} {name!r} has no property {prop!r}")
            elif not text:
                messages.append(
                    f"{kind} {name!r}: property {prop!r} has a blank description"
                )
    return messages


def _spec(entries: Mapping[str, Entry], keep: Container[str]) -> dict[str, Any]:
    return {
        name: {
            **({} if entry.description is None else {"description": entry.description}),
            "properties": dict(entry.properties),
        }
        for name, entry in entries.items()
        if name in keep
    }


def _entries(section: object, where: str) -> dict[str, Entry]:
    entries = {}
    for name, body in _mapping(section, where).items():
        label = f"{where}[{name!r}]"
        _name(name, where)
        body = _mapping(body, label)
        _only_keys(body, _ENTRY_KEYS, label)
        properties = {}
        for prop, text in _mapping(
            body.get("properties", {}), f"{label}['properties']"
        ).items():
            _name(prop, f"{label}['properties']")
            properties[prop] = _text(text, f"{label}['properties'][{prop!r}]")
        description = (
            _text(body["description"], f"{label}['description']")
            if "description" in body
            else None
        )
        entries[name] = Entry(description, properties)
    return entries


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping, not {type(value).__name__}")
    return value


def _name(name: object, where: str) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"a name in {where} must be a string, not {type(name).__name__}"
        )


def _only_keys(
    mapping: Mapping[str, Any], allowed: tuple[str, ...], where: str
) -> None:
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise ValueError(
            f"{where} has {', '.join(map(repr, unknown))}; "
            f"the keys are {', '.join(map(repr, allowed))}"
        )


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string, not {type(value).__name__}")
    return " ".join(value.split())
