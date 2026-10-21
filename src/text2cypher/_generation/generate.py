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
from text2cypher._generation.prompts import SkillsOffer, generator_system
from text2cypher._generation.read_skill import chat_reading_skills
from text2cypher.llm import AsyncChatClient, AsyncLLMClient, Completion, Message, Usage
from text2cypher.llm._base import supports_tools
from text2cypher.schema import GraphSchema


@dataclass(frozen=True, slots=True)
class Generation:
    """A query an LLM wrote, empty if it declined, and the tokens that cost.

    ``empty`` is a reply with no text at all, which is not the same as a decline (an empty
    fenced block): it is a call that returned nothing. A refusal is not empty: it is the
    model declining, in the one way it could. ``finish_reason`` is why the reply ended, when
    the client says.
    """

    query: str
    usage: Usage | None
    skills: tuple[str, ...] = ()
    empty: bool = False
    finish_reason: str | None = None


async def generate_cypher(
    llm: AsyncChatClient | AsyncLLMClient,
    schema: GraphSchema,
    question: str,
    rounds: Sequence[Round] = (),
    *,
    parameterize: bool = False,
    skills: bool = False,
) -> Generation:
    """A query for ``question`` over ``schema``; empty if the LLM says it has none.

    ``rounds`` are the queries it wrote before that had to be corrected, oldest first.
    Each correction uses the same system prompt as the first query. A client that has
    ``chat`` is given the conversation as turns; one with only ``complete`` is given it
    as one message.

    With ``skills``, a client that can be offered tools lists the skills and may read the
    ones it needs (``Generation.skills`` says which, and the skills it read before are in
    the prompt already); any other client has every skill in the prompt.

    Raises:
        SkillLoopFailed: If a call fails after others of the same request succeeded.
    """
    offer = None
    if skills:
        read = tuple(dict.fromkeys(i for round_ in rounds for i in round_.skills_read))
        offer = SkillsOffer(supports_tools(llm), read)
    system = generator_system(schema, parameterize=parameterize, skills=offer)
    messages = chat_messages(question, rounds, schema, parameterize=parameterize)
    if offer is not None and offer.tools:
        completion, read_now, usage = await chat_reading_skills(
            cast("AsyncChatClient", llm), system, messages
        )
        return Generation(
            extract_cypher(completion.text),
            usage,
            read_now,
            empty=_no_text(completion),
            finish_reason=completion.finish_reason,
        )
    completion = await ask(
        llm,
        system,
        messages,
        folded_message(question, rounds, schema, parameterize=parameterize),
    )
    return Generation(
        extract_cypher(completion.text),
        completion.usage,
        empty=_no_text(completion),
        finish_reason=completion.finish_reason,
    )


def _no_text(completion: Completion) -> bool:
    """A reply with nothing in it, that is not a refusal (which is a decline)."""
    return not completion.text.strip() and completion.finish_reason != "refusal"


async def ask(
    llm: AsyncChatClient | AsyncLLMClient,
    system: str,
    messages: Sequence[Message],
    folded: str,
) -> Completion:
    """The model's reply to a conversation: as turns when the client has ``chat``, else
    as the one message ``folded``, which says the same."""
    chat = getattr(llm, "chat", None)
    if callable(chat):
        return await chat(system, messages)
    return await cast("AsyncLLMClient", llm).complete(system, folded)
