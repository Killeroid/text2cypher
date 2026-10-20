"""Ask an LLM for a Cypher query, or to correct one."""

from collections.abc import Iterable
from dataclasses import dataclass

from text2cypher._generation.extract import extract_cypher
from text2cypher._generation.prompts import (
    corrector_user,
    generator_system,
    generator_user,
)
from text2cypher._validation import Issue
from text2cypher.llm import AsyncLLMClient, Usage
from text2cypher.schema import GraphSchema


@dataclass(frozen=True, slots=True)
class Generation:
    """A query an LLM wrote, empty if it declined, and the tokens that cost."""

    query: str
    usage: Usage | None


async def generate_cypher(
    llm: AsyncLLMClient,
    schema: GraphSchema,
    question: str,
    *,
    parameterize: bool = False,
) -> Generation:
    """A query for ``question`` over ``schema``; empty if the LLM says it has none."""
    return await _ask(
        llm,
        generator_system(schema, parameterize=parameterize),
        generator_user(question),
    )


async def correct_cypher(
    llm: AsyncLLMClient,
    schema: GraphSchema,
    question: str,
    query: str,
    issues: Iterable[Issue],
    *,
    parameterize: bool = False,
) -> Generation:
    """A corrected ``query``; empty if the LLM says the question cannot be answered.

    A correction uses the same system prompt as the first query; only the user
    message differs.
    """
    return await _ask(
        llm,
        generator_system(schema, parameterize=parameterize),
        corrector_user(question, query, issues, schema, parameterize=parameterize),
    )


async def _ask(llm: AsyncLLMClient, system: str, user: str) -> Generation:
    completion = await llm.complete(system, user)
    return Generation(extract_cypher(completion.text), completion.usage)
