"""The public API is what the package says it is, and nothing leaks into it by accident."""

import importlib
import pkgutil

import text2cypher
from text2cypher import llm, schema, skills

PUBLIC = {
    "text2cypher": {
        "AsyncChatClient",
        "AsyncLLMClient",
        "AsyncSchemaFilter",
        "Attempt",
        "Completion",
        "Config",
        "DataDictionary",
        "DatabaseError",
        "FakeLLM",
        "FilterResult",
        "GraphNode",
        "GraphNotFoundError",
        "GraphPath",
        "GraphRelationship",
        "GraphSchema",
        "Issue",
        "LLMError",
        "LLMSchemaFilter",
        "Message",
        "ModelOptions",
        "Records",
        "Result",
        "SchemaFilter",
        "Skill",
        "Text2Cypher",
        "Text2CypherError",
        "Tool",
        "ToolCall",
        "Usage",
        "ValidationResult",
        "__version__",
        "filter_schema",
        "to_python",
    },
    "text2cypher.llm": {
        "AsyncChatClient",
        "AsyncLLMClient",
        "Call",
        "Completion",
        "FakeLLM",
        "Message",
        "ModelOptions",
        "Tool",
        "ToolCall",
        "Usage",
    },
    "text2cypher.skills": {"Skill", "builtin_skills", "load_skills"},
    "text2cypher.schema": {
        "DataDictionary",
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
        "ModelOptions",
        "Tool",
        "ToolCall",
        "Usage",
    ):
        assert getattr(text2cypher, name) is getattr(llm, name)
    for name in ("DataDictionary", "GraphSchema", "SchemaFilter", "filter_schema"):
        assert getattr(text2cypher, name) is getattr(schema, name)
    assert text2cypher.Skill is skills.Skill


def test_every_module_outside_the_public_namespaces_is_internal():
    public_packages = {
        "text2cypher",
        "text2cypher.llm",
        "text2cypher.schema",
        "text2cypher.skills",
    }
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


def test_the_adapter_of_the_sdk_client_is_not_public_and_the_old_name_is_gone():
    # The SDK's own client is given to Text2Cypher with model= and model_options=
    for module in (text2cypher, llm):
        assert not hasattr(module, "OpenAIClient")
        assert not hasattr(module, "ChatCompletionsAdapter")
