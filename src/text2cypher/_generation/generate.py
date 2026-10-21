"""Ask an LLM for a Cypher query, or to correct one."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from text2cypher._generation.conversation import (
    Round,
    chat_messages,
    folded_message,
)
from text2cypher._generation.extract import extract_cypher
from text2cypher._generation.prompts import generator_system
from text2cypher.llm import AsyncChatClient, AsyncLLMClient, Completion, Usage
from text2cypher.schema import GraphSchema


@dataclass(frozen=True, slots=True)
class Generation:
    """A query an LLM wrote, empty if it declined, and the tokens that cost."""

    query: str
    usage: Usage | None


async def generate_cypher(
    llm: AsyncChatClient | AsyncLLMClient,
    schema: GraphSchema,
    question: str,
    rounds: Sequence[Round] = (),
    *,
    parameterize: bool = False,
) -> Generation:
    """A query for ``question`` over ``schema``; empty if the LLM says it has none.

    ``rounds`` are the queries it wrote before that had to be corrected, oldest first.
    Each correction uses the same system prompt as the first query. A client that has
    ``chat`` is given the conversation as turns; one with only ``complete`` is given it
    as one message.
    """
    system = generator_system(schema, parameterize=parameterize)
    chat = getattr(llm, "chat", None)
    if callable(chat):
        completion: Completion = await chat(
            system, chat_messages(question, rounds, schema, parameterize=parameterize)
        )
    else:
        completion = await cast("AsyncLLMClient", llm).complete(
            system, folded_message(question, rounds, schema, parameterize=parameterize)
        )
    return Generation(extract_cypher(completion.text), completion.usage)
