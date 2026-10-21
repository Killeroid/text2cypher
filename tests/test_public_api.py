"""The public API is what the package says it is, and nothing leaks into it by accident."""

import importlib
import pkgutil

import text2cypher
from text2cypher import llm, schema

PUBLIC = {
    "text2cypher": {
        "AsyncChatClient",
        "AsyncLLMClient",
        "AsyncSchemaFilter",
        "Attempt",
        "Completion",
        "Config",
        "DatabaseError",
        "FakeLLM",
        "FilterResult",
        "GraphNotFoundError",
        "GraphSchema",
        "Issue",
        "LLMError",
        "LLMSchemaFilter",
        "Message",
        "OpenAIClient",
        "Records",
        "Result",
        "SchemaFilter",
        "Text2Cypher",
        "Text2CypherError",
        "Usage",
        "ValidationResult",
        "__version__",
        "filter_schema",
    },
    "text2cypher.llm": {
        "AsyncChatClient",
        "AsyncLLMClient",
        "Call",
        "Completion",
        "FakeLLM",
        "Message",
        "OpenAIClient",
        "Usage",
    },
    "text2cypher.schema": {
        "GraphSchema",
        "Index",
        "Path",
        "SchemaFilter",
        "filter_schema",
        "render_schema",
        "restrict_schema",
        "schema_from_snapshot",
    },
}


def test_each_public_namespace_exports_exactly_its_listed_names():
    for name, expected in PUBLIC.items():
        module = importlib.import_module(name)
        assert set(module.__all__) == expected, name
        assert all(hasattr(module, member) for member in expected), name


def test_the_public_names_of_the_subpackages_are_the_same_objects_as_the_top_level_ones():
    for name in (
        "AsyncChatClient",
        "AsyncLLMClient",
        "Completion",
        "FakeLLM",
        "Message",
        "OpenAIClient",
        "Usage",
    ):
        assert getattr(text2cypher, name) is getattr(llm, name)
    for name in ("GraphSchema", "SchemaFilter", "filter_schema"):
        assert getattr(text2cypher, name) is getattr(schema, name)


def test_every_module_outside_the_public_namespaces_is_internal():
    public_packages = {"text2cypher", "text2cypher.llm", "text2cypher.schema"}
    names = [
        module.name
        for module in pkgutil.walk_packages(text2cypher.__path__, "text2cypher.")
    ]
    exposed = [
        name
        for name in names
        if name not in public_packages
        and not any(part.startswith("_") for part in name.split("."))
    ]
    assert exposed == []


def test_the_version_is_a_string():
    assert isinstance(text2cypher.__version__, str)
    assert text2cypher.__version__
