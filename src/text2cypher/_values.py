"""Nodes, relationships and paths as plain Python objects, with no client in them."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from falkordb.edge import Edge
from falkordb.node import Node
from falkordb.path import Path


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class GraphNode:
    """A node a query returned, as a plain object.

    Two nodes are equal when they have the same ``id``, and the same hash.

    Attributes:
        id: The node's id in the graph.
        labels: Its labels.
        properties: Its properties, read-only. ``node["name"]`` and ``node.get("name")``
            read one.
    """

    id: int
    labels: tuple[str, ...]
    properties: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    def __getitem__(self, name: str) -> Any:
        return self.properties[name]

    def get(self, name: str, default: Any = None) -> Any:
        """The property ``name``, or ``default`` if there is none."""
        return self.properties.get(name, default)

    def __reduce__(self) -> tuple[Any, ...]:
        return (GraphNode, (self.id, self.labels, dict(self.properties)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphNode) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("node", self.id))

    def __repr__(self) -> str:
        return f"GraphNode(id={self.id!r}, labels={self.labels!r}, properties={dict(self.properties)!r})"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class GraphRelationship:
    """A relationship a query returned, as a plain object.

    Two relationships are equal when they have the same ``id``, and the same hash.

    Attributes:
        id: The relationship's id in the graph.
        type: Its type, for example ``"KNOWS"``.
        start_id: The id of the node it starts at.
        end_id: The id of the node it ends at.
        properties: Its properties, read-only.
    """

    id: int
    type: str
    start_id: int
    end_id: int
    properties: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))

    def __getitem__(self, name: str) -> Any:
        return self.properties[name]

    def get(self, name: str, default: Any = None) -> Any:
        """The property ``name``, or ``default`` if there is none."""
        return self.properties.get(name, default)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            GraphRelationship,
            (self.id, self.type, self.start_id, self.end_id, dict(self.properties)),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphRelationship) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("relationship", self.id))

    def __repr__(self) -> str:
        return (
            f"GraphRelationship(id={self.id!r}, type={self.type!r}, start_id={self.start_id!r}, "
            f"end_id={self.end_id!r}, properties={dict(self.properties)!r})"
        )


@dataclass(frozen=True, slots=True)
class GraphPath:
    """A path a query returned, as plain objects.

    Attributes:
        nodes: The nodes along the path, in order.
        relationships: The relationships between them, in order: one fewer than the nodes.
    """

    nodes: tuple[GraphNode, ...]
    relationships: tuple[GraphRelationship, ...]


def to_python(value: Any) -> Any:
    """``value`` with every node, relationship and path in it as a plain object.

    A ``falkordb`` node, edge or path becomes a ``GraphNode``, ``GraphRelationship`` or
    ``GraphPath``; a list, tuple or map is converted inside, and a map comes back as a
    ``dict``. Anything else (text, numbers, dates, points) is the client's own value,
    which is already plain. A value that was converted before is returned as it is.
    """
    if isinstance(value, Node):
        return GraphNode(_id(value), tuple(value.labels or ()), value.properties or {})
    if isinstance(value, Edge):
        return GraphRelationship(
            _id(value),
            value.relation,
            _id(value.src_node),
            _id(value.dest_node),
            value.properties or {},
        )
    if isinstance(value, Path):
        return GraphPath(
            tuple(to_python(node) for node in value.nodes()),
            tuple(to_python(edge) for edge in value.edges()),
        )
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_python(item) for item in value)
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    return value


def _id(entity: Any) -> int:
    """The id of a node or relationship, or an id that is already one."""
    return getattr(entity, "id", entity)
