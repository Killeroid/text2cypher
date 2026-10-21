"""The shape of a graph, as the LLM is shown it."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

_PLAIN_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Path(NamedTuple):
    """A one-hop path that exists in the graph: ``(:source)-[:type]->(:target)``.

    Attributes:
        source: The label of the node the relationship starts at.
        type: The relationship type.
        target: The label of the node it ends at.
    """

    source: str
    type: str
    target: str

    def pattern(self) -> str:
        """The path written as Cypher, with names quoted where they need it."""
        return (
            f"(:{quote_identifier(self.source)})"
            f"-[:{quote_identifier(self.type)}]->"
            f"(:{quote_identifier(self.target)})"
        )


class Index(NamedTuple):
    """An index on one property of a label or relationship type.

    Attributes:
        entity: The label or relationship type the index is on.
        property: The indexed property.
        type: The kind of index FalkorDB reports: ``"RANGE"``, ``"FULLTEXT"`` or ``"VECTOR"``.
    """

    entity: str
    property: str
    type: str


@dataclass(frozen=True, slots=True)
class GraphSchema:
    """What a graph contains, found from its data (FalkorDB has no declared schema).

    ``node_properties`` maps each node label to its properties, and
    ``relationship_properties`` each relationship type to its properties. A property
    maps to the value types seen for it, e.g. ``{"year": ("Integer",)}``.

    Attributes:
        node_properties: The properties of each label, with their value types.
        relationship_properties: The properties of each relationship type.
        paths: The one-hop paths that exist.
        indexes: The indexes on properties of labels and relationship types.
        functions: The user-defined functions of the FalkorDB instance, as ``library.function``.
        node_examples: Example values of the properties of each label, as they are written in
            a query (``'Alien'``, ``1999``): at most a few per property, and only for some.
        relationship_examples: The same for the properties of each relationship type.
    """

    node_properties: Mapping[str, Mapping[str, tuple[str, ...]]]
    relationship_properties: Mapping[str, Mapping[str, tuple[str, ...]]]
    paths: tuple[Path, ...]
    indexes: tuple[Index, ...] = ()
    functions: tuple[str, ...] = ()
    node_examples: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    relationship_examples: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )


def quote_identifier(name: str) -> str:
    """Quote a label, relationship type or property name for use in Cypher, if needed.

    Raises:
        ValueError: If ``name`` contains a backtick. FalkorDB cannot parse an
            escaped backtick, so such a name cannot be written in a query.
    """
    if _PLAIN_IDENTIFIER.fullmatch(name):
        return name
    if "`" in name:
        raise ValueError(
            f"Cannot quote {name!r}: FalkorDB has no escape for a backtick"
        )
    return f"`{name}`"
