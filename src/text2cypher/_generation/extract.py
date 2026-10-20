"""Get the Cypher query out of what an LLM replied."""

import re

# The opening fence, then an optional language tag, then the query. A tag is any word on the
# fence's own line, or a Cypher tag followed by a space on a one-line block.
_FENCE = re.compile(
    r"```[ \t]*(?:[A-Za-z]+[ \t]*\r?\n|(?:cypher|opencypher|cql)(?![A-Za-z0-9_])[ \t]*)?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_cypher(reply: str) -> str:
    """The query in the first fenced block of ``reply``, else ``reply`` trimmed.

    Models often add prose around the fence, so the first block is the answer
    wherever it is. An empty block gives an empty query. A reply cut off before its
    closing fence is returned as it is, and fails validation like any other bad query.
    """
    if match := _FENCE.search(reply):
        return match.group(1).strip()
    return reply.strip()
