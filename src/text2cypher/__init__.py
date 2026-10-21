"""Turn a question into a read-only Cypher query that passes validation against a FalkorDB graph.

The public API is what is listed in ``__all__`` here and in ``text2cypher.llm`` and
``text2cypher.schema``. Every module whose name starts with an underscore is internal and may
change without notice.
"""

from importlib.metadata import PackageNotFoundError, version

from text2cypher._config import Config
from text2cypher._errors import GraphNotFoundError, Text2CypherError
from text2cypher._generation import AsyncSchemaFilter, FilterResult, LLMSchemaFilter
from text2cypher._pipeline import Text2Cypher
from text2cypher._results import (
    Attempt,
    DatabaseError,
    LLMError,
    Records,
    Result,
)
from text2cypher._validation import Issue, ValidationResult
from text2cypher.llm import (
    AsyncChatClient,
    AsyncLLMClient,
    Completion,
    FakeLLM,
    Message,
    OpenAIClient,
    Usage,
)
from text2cypher.schema import GraphSchema, SchemaFilter, filter_schema

try:
    __version__ = version("text2cypher")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
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
]
