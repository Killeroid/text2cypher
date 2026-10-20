"""An LLM client for OpenAI's Chat Completions API (and compatible servers)."""

from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionMessageParam

from text2cypher.llm._base import Completion, Usage


class OpenAIClient:
    """Asks an OpenAI chat model, one system and one user message per call.

    A model that has no system role (some served by compatible servers) needs
    ``system_role=False``: the system text then goes first in the one user message.

    Args:
        model: The model to ask. There is no default: the right one depends on the
            account and on the schema being prompted.
        client: The ``AsyncOpenAI`` to use, e.g. one with another ``base_url`` for a
            compatible server. By default one is made from the environment.
        temperature: ``None`` leaves it out, for models that only accept their own.
        system_role: ``False`` sends one user message, the system text followed by the
            user text, for a model that has no system role.
    """

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI | None = None,
        temperature: float | None = 0,
        system_role: bool = True,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI()
        self._temperature = omit if temperature is None else temperature
        self._system_role = system_role

    async def complete(self, system: str, user: str) -> Completion:
        """Ask the model, and return its reply and, when the API reports it, the tokens used."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(system, user),
            temperature=self._temperature,
        )
        used = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            usage=None
            if used is None
            else Usage(used.prompt_tokens, used.completion_tokens),
        )

    def _messages(self, system: str, user: str) -> list[ChatCompletionMessageParam]:
        if self._system_role:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
