"""Build a ``GraphSchema`` from a cypher-guardrail ``Snapshot``."""

from collections.abc import Iterable
from typing import Any

from cypher_guardrail import Snapshot

from text2cypher.schema._model import GraphSchema, Index, Path


def schema_from_snapshot(snapshot: Snapshot) -> GraphSchema:
    """The labels, relationship types, properties, paths and indexes recorded in ``snapshot``.

    Capture the snapshot with ``types=True``: the value types of each property come
    from it. The snapshot records an edge between sets of labels, so a node with
    several labels gives one path per label pair.

    Labels, relationship types and properties whose name contains a backtick are
    left out: FalkorDB cannot parse an escaped backtick, so no query can use them,
    and quoting them is not possible.

    Raises:
        ValueError: If the snapshot was captured without ``types=True``.
    """
    if snapshot.property_types is None:
        raise ValueError("The snapshot has no value types: capture it with types=True")
    node_properties, relationship_properties = _properties(snapshot.property_types)
    return GraphSchema(
        node_properties=node_properties,
        relationship_properties=relationship_properties,
        paths=_paths(snapshot.edges),
        indexes=_indexes(snapshot.indexes),
    )


type _Properties = dict[str, dict[str, tuple[str, ...]]]


def _properties(property_types: Iterable[Any]) -> tuple[_Properties, _Properties]:
    """The properties of each label and of each relationship type, with their value types."""
    nodes: _Properties = {}
    relationships: _Properties = {}
    for name, kind, properties in property_types:
        if _usable(name):
            (nodes if kind == "NODE" else relationships)[name] = {
                property_: tuple(value_types)
                for property_, value_types in properties
                if _usable(property_)
            }
    return nodes, relationships


def _paths(edges: Iterable[Any]) -> tuple[Path, ...]:
    """One path per pair of labels at the two ends of each edge."""
    return tuple(
        sorted(
            {
                Path(source, type_, target)
                for sources, type_, targets in edges
                if _usable(type_)
                for source in filter(_usable, sources)
                for target in filter(_usable, targets)
            }
        )
    )


def _indexes(indexes: Iterable[Any]) -> tuple[Index, ...]:
    return tuple(
        sorted(
            {
                Index(entity, attribute, index_type)
                for entity, _, attributes in indexes
                if _usable(entity)
                for attribute, index_types in attributes
                if _usable(attribute)
                for index_type in index_types
            }
        )
    )


def _usable(name: str) -> bool:
    return "`" not in name
