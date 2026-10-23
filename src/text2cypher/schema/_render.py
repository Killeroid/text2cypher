"""Turn a schema into the text an LLM is prompted with."""

from collections.abc import Collection, Mapping
from typing import NamedTuple

from text2cypher.schema._dictionary import Entry
from text2cypher.schema._model import GraphSchema, Index, quote_identifier


class Section(NamedTuple):
    """A part of the rendered schema: a title and what is under it.

    ``body`` is the text under the title; a section with ``children`` has them in its place.
    """

    title: str
    body: str
    children: tuple["Section", ...] = ()


def render_schema(schema: GraphSchema) -> str:
    """Describe ``schema`` as node properties, relationship properties, paths, indexes,
    functions and the data dictionary.

    Sections with nothing to say are left out, so an empty schema renders as "".
    """
    return "\n".join(map(as_text, schema_sections(schema)))


def as_text(section: Section) -> str:
    """A section as plain text: its title and a colon, then its body or its children."""
    inside = section.body or "\n".join(map(as_text, section.children))
    return f"{section.title}:\n{inside}"


# The title of a section as a Markdown heading, where it differs from the plain one
_HEADINGS = {"The relationships": "Relationships"}


def as_markdown(section: Section, level: int) -> str:
    """A section as Markdown: a heading of ``level``, then its body or its children.

    The children are written one level down.
    """
    title = _HEADINGS.get(section.title, section.title)
    inside = (
        "\n".join(as_markdown(child, level + 1) for child in section.children)
        if section.children
        else section.body
    )
    return f"{'#' * level} {title}\n{inside}"


def schema_sections(schema: GraphSchema) -> list[Section]:
    """The parts ``render_schema`` writes, in order, for a prompt that lays them out itself.

    A part with ``children`` (the data dictionary) has them in place of a body.
    """
    sections = []
    if schema.node_properties:
        sections.append(
            Section(
                "Node properties",
                _entities(
                    schema.node_properties, schema.node_examples, keep_empty=True
                ),
            )
        )
    # A relationship type without properties still appears in the paths
    if any(schema.relationship_properties.values()):
        sections.append(
            Section(
                "Relationship properties",
                _entities(
                    schema.relationship_properties,
                    schema.relationship_examples,
                    keep_empty=False,
                ),
            )
        )
    if schema.paths:
        sections.append(
            Section(
                "The relationships",
                "\n".join(path.pattern() for path in schema.paths),
            )
        )
    if schema.indexes:
        sections.append(Section("Indexes", _indexes(schema.indexes)))
    if schema.functions:
        sections.append(Section("User-defined functions", "\n".join(schema.functions)))
    if dictionary := data_dictionary_section(schema):
        sections.append(dictionary)
    return sections


def data_dictionary_section(
    schema: GraphSchema, *, properties: bool = True
) -> Section | None:
    """The data dictionary as a section with a child for each kind of name, or ``None``.

    ``None`` when there is no dictionary, or when it has no text for what is shown: with
    ``properties=False`` only the descriptions of labels and relationship types count and
    only they are listed.
    """
    dictionary = schema.data_dictionary
    if dictionary is None:
        return None
    types = dict.fromkeys(
        (*schema.relationship_properties, *(path.type for path in schema.paths))
    )
    kinds = (
        ("Labels", dictionary.nodes, schema.node_properties),
        (
            "Relationship types",
            dictionary.relationships,
            {name: schema.relationship_properties.get(name, {}) for name in types},
        ),
    )
    if not any(
        _has_text(entries.get(name), own, properties)
        for _, entries, known in kinds
        for name, own in known.items()
    ):
        return None
    # A heading says which kind of name each line is: a label and a relationship type can
    # have the same name
    children = tuple(
        Section(title, "\n".join(_entities_listed(entries, known, properties)))
        for title, entries, known in kinds
        if known
    )
    return Section("Data dictionary", "", children)


def _has_text(entry: Entry | None, own: Collection[str], properties: bool) -> bool:
    """Whether ``entry`` says something about what the graph has.

    That is its description, or, if ``properties`` are shown, the text of one of ``own``.
    """
    if entry is None:
        return False
    return bool(entry.description) or (
        properties and any(entry.properties.get(name) for name in own)
    )


def _entities_listed(
    entries: Mapping[str, Entry],
    known: Mapping[str, Mapping[str, tuple[str, ...]]],
    properties: bool,
) -> list[str]:
    """The list items of one kind of name: an entity, then under it its properties.

    Every entity the graph has is listed, with its description if it has one, and every
    property of it, with its text if it has some. The items have no styling, so that a name
    looks as it does in the schema.
    """
    lines = []
    for name, own in known.items():
        entry = entries.get(name)
        lines.append(_item(name, entry.description if entry else None, 0))
        if properties:
            said = entry.properties if entry else {}
            lines.extend(_item(prop, said.get(prop), 1) for prop in own)
    return lines


def _item(name: str, text: str | None, depth: int) -> str:
    indent = "  " * depth
    return f"{indent}- {quote_identifier(name)}" + (f": {text}" if text else "")


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
