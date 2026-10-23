"""Accept the SDK's own client wherever an LLM is accepted."""

from openai import AsyncOpenAI, OpenAI

from text2cypher.llm._base import AsyncChatClient, AsyncLLMClient
from text2cypher.llm._chat_completions import ChatCompletionsAdapter
from text2cypher.llm._options import ModelOptions

type LLM = AsyncChatClient | AsyncLLMClient | AsyncOpenAI


def as_client(
    llm: LLM,
    model: str | None = None,
    model_options: ModelOptions | None = None,
) -> AsyncChatClient | AsyncLLMClient:
    """``llm`` as a client the library can ask.

    An ``openai.AsyncOpenAI`` (also ``AsyncAzureOpenAI``, and one made for a compatible
    server with ``base_url``) becomes a ``ChatCompletionsAdapter`` for ``model``, with
    ``model_options`` or else that class's defaults. Any other client is returned as it is.

    Raises:
        ValueError: If an ``AsyncOpenAI`` is given with no ``model`` (it has none of its
            own), or ``model`` or ``model_options`` is given with a client that is not one.
        TypeError: If ``llm`` is the synchronous ``openai.OpenAI``: the library is async.
    """
    if isinstance(llm, AsyncOpenAI):
        if model is None:
            raise ValueError(
                "An openai.AsyncOpenAI has no model of its own: pass model='...'"
            )
        return ChatCompletionsAdapter(model, client=llm, model_options=model_options)
    if isinstance(llm, OpenAI):
        raise TypeError(
            "text2cypher is async: pass an openai.AsyncOpenAI, not an openai.OpenAI"
        )
    if model is not None:
        raise ValueError(
            "model= is for an openai.AsyncOpenAI; any other client chooses its own model"
        )
    if model_options is not None:
        raise ValueError(
            "model_options is for an openai.AsyncOpenAI; any other client has its own "
            "settings, which it is given when it is made"
        )
    return llm
