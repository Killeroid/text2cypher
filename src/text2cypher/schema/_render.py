"""Turn a schema into the text an LLM is prompted with."""

from collections.abc import Mapping

from text2cypher.schema._model import GraphSchema, Index, quote_identifier


def render_schema(schema: GraphSchema) -> str:
    """Describe ``schema`` as node properties, relationship properties, paths, indexes and functions.

    Sections with nothing to say are left out, so an empty schema renders as "".
    """
    sections = []
    if schema.node_properties:
        sections.append(
            "Node properties:\n"
            + _entities(schema.node_properties, schema.node_examples, keep_empty=True)
        )
    # A relationship type without properties still appears in the paths
    if any(schema.relationship_properties.values()):
        sections.append(
            "Relationship properties:\n"
            + _entities(
                schema.relationship_properties,
                schema.relationship_examples,
                keep_empty=False,
            )
        )
    if schema.paths:
        sections.append(
            "The relationships:\n" + "\n".join(path.pattern() for path in schema.paths)
        )
    if schema.indexes:
        sections.append("Indexes:\n" + _indexes(schema.indexes))
    if schema.functions:
        sections.append("User-defined functions:\n" + "\n".join(schema.functions))
    return "\n".join(sections)


def _indexes(indexes: tuple[Index, ...]) -> str:
    """One line per indexed property, with the kinds of index it has."""
    by_property: dict[tuple[str, str], list[str]] = {}
    for entity, prop, index_type in sorted(indexes):
        by_property.setdefault((entity, prop), []).append(index_type)
    return "\n".join(
        f"{quote_identifier(entity)}.{quote_identifier(prop)}: {', '.join(types)}"
        for (entity, prop), types in by_property.items()
    )


def _entities(
    entities: Mapping[str, Mapping[str, tuple[str, ...]]],
    examples: Mapping[str, Mapping[str, tuple[str, ...]]],
    *,
    keep_empty: bool,
) -> str:
    lines = []
    for name, properties in entities.items():
        if properties or keep_empty:
            body = ", ".join(
                _property(prop, types, examples.get(name, {}).get(prop, ()))
                for prop, types in properties.items()
            )
            lines.append(f"{quote_identifier(name)} {{{body}}}")
    return "\n".join(lines)


def _property(name: str, types: tuple[str, ...], examples: tuple[str, ...]) -> str:
    text = f"{quote_identifier(name)}: {'|'.join(t.upper() for t in types)}"
    return f"{text} (e.g. {', '.join(examples)})" if examples else text
