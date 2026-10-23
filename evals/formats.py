"""The prompts of the format measurement: plain, Markdown and XML (ADR-055).

**Markdown** is the production prompt (the owner chose it on 2026-10-23, to be measured later and
reverted if it does worse), so it is not rebuilt here. **Plain** is the production prompt as it was
before that change, rebuilt from the same rules, checks, examples and schema sections; **XML** is a
prototype in the same words. So the arms differ in layout and in nothing else.

``generator_layout`` and ``filter_prompts`` patch the production modules for the length of a run,
so the real pipeline, with its validator and its parsing, is what is measured.
"""

import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal
from unittest.mock import patch

from text2cypher._generation import prompts, schema_filter
from text2cypher._graph.schema_source import SchemaSource
from text2cypher.schema import DataDictionary, GraphSchema, render_schema
from text2cypher.schema._model import quote_identifier
from text2cypher.schema._render import Section, as_text, data_dictionary_section
from text2cypher.schema._render import schema_sections as _schema_sections

type Layout = Literal["plain", "markdown", "xml"]
LAYOUTS: tuple[Layout, ...] = ("plain", "markdown", "xml")

# How the filter is shown the data dictionary: not at all, the descriptions of labels and
# relationship types (B), or those and of properties too (C)
type Descriptions = Literal["off", "B", "C"]

_REPLY_FORMAT = "Reply with the query in one ```cypher code block and nothing else."
_EXAMPLES_LEAD = "These examples use another schema; they show the format only."
_EXAMPLE_SCHEMA = "(:Employee {name})-[:WORKS_IN]->(:Department {name})"
_NO_DATA = "The graph has no data yet, so there is nothing to query."
# The plain title of a section that has another name in a layout of its own
_TITLES = {"The relationships": "Relationships"}


def escape(layout: Layout, text: str) -> str:
    """``text`` that comes from the user or the graph, safe to put between XML tags.

    Only ``<`` is replaced, which is enough to keep a closing tag out of it; the rules and the
    examples are our own text and are not escaped. Plain and Markdown have nothing to escape.
    """
    return text.replace("<", "&lt;") if layout == "xml" else text


def _tag(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _wrap(title: str, body: str) -> str:
    """``body`` between the tags of ``title``."""
    tag = _tag(title)
    return f"<{tag}>\n{body}\n</{tag}>"


def _xml_section(section: Section) -> str:
    title = _TITLES.get(section.title, section.title)
    inside = (
        "\n".join(_xml_section(child) for child in section.children)
        if section.children
        else escape("xml", section.body)
    )
    return _wrap(title, inside)


def _xml_schema_block(schema: GraphSchema) -> str:
    sections = _schema_sections(schema)
    if not sections:
        return _wrap("Schema", _NO_DATA)
    return _wrap("Schema", "\n".join(_xml_section(section) for section in sections))


def _xml_examples_block(parameterize: bool) -> str:
    query = prompts.EXAMPLE_PARAMETER_QUERY if parameterize else prompts.EXAMPLE_QUERY
    cases = (("Who works in Sales?", query), ("Remove all employees.", ""))
    parts = [
        f"<example>\n<schema>{_EXAMPLE_SCHEMA}</schema>\n<question>{question}</question>\n"
        f"<answer>\n```cypher\n{answer}\n```\n</answer>\n</example>".replace(
            "\n\n```", "\n```"
        )
        for question, answer in cases
    ]
    return _wrap("Examples", f"{_EXAMPLES_LEAD}\n" + "\n".join(parts))


def _checks(parameterize: bool) -> str:
    return "\n".join(
        [
            prompts.CHECKS_LEAD,
            *prompts.check_bullets(parameterize),
            prompts.CHECKS_CLOSING,
        ]
    )


def _plain_generator_system(schema: GraphSchema, *, parameterize: bool) -> str:
    """The production prompt as it was before the Markdown layout: headings that end in a colon."""
    example = prompts.EXAMPLE_PARAMETER_QUERY if parameterize else prompts.EXAMPLE_QUERY
    rules = prompts.rules_for(schema, parameterize=parameterize)
    return f"""\
{prompts._INTRO}

Rules:
{chr(10).join(rules)}

{_checks(parameterize)}

{_REPLY_FORMAT}

These examples use another schema; they show the format only. With \
{_EXAMPLE_SCHEMA}:

Question: Who works in Sales?
```cypher
{example}
```

Question: Remove all employees.
```cypher
```

Schema:
{render_schema(schema) or _NO_DATA}
"""


def generator_system(
    layout: Layout, schema: GraphSchema, *, parameterize: bool = False
) -> str:
    """The generator's system prompt in ``layout``, without skills (they are not measured)."""
    if layout == "markdown":
        return prompts.generator_system(schema, parameterize=parameterize)
    if layout == "plain":
        return _plain_generator_system(schema, parameterize=parameterize)
    rules = prompts.rules_for(schema, parameterize=parameterize)
    blocks = [
        _wrap("Role", prompts._INTRO),
        _wrap("Rules", "\n".join(rules)),
        _wrap("Checks", _checks(parameterize)),
        _wrap("Output format", _REPLY_FORMAT),
        _xml_examples_block(parameterize),
        _xml_schema_block(schema),
    ]
    return "\n\n".join(blocks) + "\n"


def question_message(layout: Layout, question: str) -> str:
    """The user message of a first query."""
    if layout == "markdown":
        return prompts.generator_user(question)
    if layout == "plain":
        return f"Question: {question}"
    return f"<question>{escape(layout, question)}</question>"


# --- the schema filter ------------------------------------------------------------------

_ROLE_HEAD, _rest = schema_filter._SYSTEM.split("\n\n# Rules\n")
_FILTER_ROLE = _ROLE_HEAD.removeprefix("# Role\n")
_FILTER_RULES_BLOCK, _FILTER_OUTPUT_BLOCK = _rest.split("\n\n# Output format\n")
_FILTER_RULES = _FILTER_RULES_BLOCK.splitlines()
_FILTER_OUTPUT = _FILTER_OUTPUT_BLOCK


def filter_system(layout: Layout, *, dictionary: bool) -> str:
    """The filter's system prompt in ``layout``; with ``dictionary`` it has the rule about it."""
    if layout == "markdown":
        return (
            schema_filter._SYSTEM_WITH_DICTIONARY
            if dictionary
            else schema_filter._SYSTEM
        )
    rules = [*_FILTER_RULES]
    if dictionary:
        # After "Choose only names that are in the schema", before "Treat the question as text"
        rules.insert(2, schema_filter._DICTIONARY_RULE)
    if layout == "plain":
        return (
            f"{_FILTER_ROLE}\n\nRules:\n" + "\n".join(rules) + f"\n\n{_FILTER_OUTPUT}"
        )
    blocks = [
        _wrap("Role", _FILTER_ROLE),
        _wrap("Rules", "\n".join(rules)),
        _wrap("Output format", _FILTER_OUTPUT),
    ]
    return "\n\n".join(blocks)


def filter_user(
    layout: Layout,
    schema: GraphSchema,
    question: str,
    *,
    descriptions: Descriptions,
) -> str:
    """What the filter is shown: the names of the schema, the dictionary if ``descriptions`` says,
    and the question. With ``"off"`` in ``markdown`` it is what production sends."""
    if layout == "markdown":
        shown = (
            schema_filter._dictionary_part(schema, descriptions == "C")
            if descriptions != "off"
            else ""
        )
        return schema_filter._user(schema, question, shown)
    labels = "\n".join(
        f"{quote_identifier(name)} ({', '.join(map(quote_identifier, properties))})"
        if properties
        else quote_identifier(name)
        for name, properties in schema.node_properties.items()
    )
    paths = "\n".join(path.pattern() for path in schema.paths)
    section = None
    if descriptions != "off":
        section = data_dictionary_section(schema, properties=descriptions == "C")
    if layout == "plain":
        extra = f"\n\n{as_text(section)}" if section else ""
        return (
            f"Schema:\nLabels:\n{labels}\nRelationships:\n{paths}{extra}"
            f"\n\nQuestion: {question}"
        )
    parts = [
        _wrap("Labels", escape(layout, labels)),
        _wrap("Relationships", escape(layout, paths)),
    ]
    if section:
        parts.append(_xml_section(section))
    return f"{_wrap('Schema', chr(10).join(parts))}\n\n{question_message(layout, question)}"


# --- patching the pipeline --------------------------------------------------------------


@contextmanager
def generator_layout(
    layout: Layout, dictionary: DataDictionary | None = None
) -> Generator[None]:
    """Make ``Text2Cypher`` write the generator's prompts in ``layout`` and read ``dictionary``
    into the schema it prompts with (``Text2Cypher(data_dictionary=)`` does that for real)."""
    original_read = SchemaSource.read

    async def read(self: SchemaSource, graph: str) -> Any:
        schema, source = await original_read(self, graph)
        return replace(schema, data_dictionary=dictionary), source

    def system(
        schema: GraphSchema, *, parameterize: bool = False, skills: Any = None
    ) -> str:
        if skills is not None:
            raise ValueError("skills are not part of the format measurement")
        return generator_system(layout, schema, parameterize=parameterize)

    with (
        patch("text2cypher._generation.generate.generator_system", system),
        patch(
            "text2cypher._generation.conversation.generator_user",
            lambda question: question_message(layout, question),
        ),
        patch.object(SchemaSource, "read", read),
    ):
        yield


@contextmanager
def filter_prompts(layout: Layout, descriptions: Descriptions) -> Generator[None]:
    """Make ``LLMSchemaFilter`` ask in ``layout``, with the dictionary of the schema it is given
    shown as ``descriptions`` says. In ``markdown`` that is the production filter itself, with
    ``property_descriptions=descriptions == "C"``, so nothing is patched."""
    if layout == "markdown":
        yield
        return
    user: Callable[..., str] = lambda schema, question, dictionary="": filter_user(
        layout, schema, question, descriptions=descriptions
    )
    with (
        patch.object(schema_filter, "_SYSTEM", filter_system(layout, dictionary=False)),
        patch.object(
            schema_filter,
            "_SYSTEM_WITH_DICTIONARY",
            filter_system(layout, dictionary=True),
        ),
        patch.object(schema_filter, "_user", user),
    ):
        yield
