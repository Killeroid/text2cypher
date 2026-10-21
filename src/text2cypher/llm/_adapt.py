"""Accept the SDK's own client wherever an LLM is accepted."""

from openai import AsyncOpenAI, OpenAI

from text2cypher.llm._base import AsyncChatClient, AsyncLLMClient
from text2cypher.llm._openai import OpenAIClient

type LLM = AsyncChatClient | AsyncLLMClient | AsyncOpenAI


def as_client(llm: LLM, model: str | None = None) -> AsyncChatClient | AsyncLLMClient:
    """``llm`` as a client the library can ask.

    An ``openai.AsyncOpenAI`` (also ``AsyncAzureOpenAI``, and one made for a compatible
    server with ``base_url``) becomes an ``OpenAIClient`` for ``model``, with that class's
    defaults; use ``OpenAIClient`` yourself for its options. Any other client is returned
    as it is.

    Raises:
        ValueError: If an ``AsyncOpenAI`` is given with no ``model`` (it has none of its
            own), or ``model`` is given with a client that is not one.
        TypeError: If ``llm`` is the synchronous ``openai.OpenAI``: the library is async.
    """
    if isinstance(llm, AsyncOpenAI):
        if model is None:
            raise ValueError(
                "An openai.AsyncOpenAI has no model of its own: pass model='...'"
            )
        return OpenAIClient(model, client=llm)
    if isinstance(llm, OpenAI):
        raise TypeError(
            "text2cypher is async: pass an openai.AsyncOpenAI, not an openai.OpenAI"
        )
    if model is not None:
        raise ValueError(
            "model= is for an openai.AsyncOpenAI; any other client chooses its own model"
        )
    return llm
