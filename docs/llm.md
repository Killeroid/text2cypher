# LLM clients

## The protocol

An LLM is anything with one method: it gets a system message and a user message, and returns the text and, if it knows them, the tokens used ({py:class}`~text2cypher.AsyncLLMClient`).

```python
from text2cypher import Completion, Text2Cypher, Usage


class MyClient:
    async def complete(self, system: str, user: str) -> Completion:
        text, tokens_in, tokens_out = await ask_my_model(system, user)
        return Completion(text, Usage(input_tokens=tokens_in, output_tokens=tokens_out))


agent = Text2Cypher(client, MyClient())
```

Each call is independent: text2cypher sends everything the model needs in the two messages and keeps no conversation. Use `Completion(text, None)` when the provider does not report tokens; `result.usage` is then `None`.

The reply should hold the query in one fenced block (` ```cypher ... ``` `). An empty block means the question cannot be answered. Text around the block is ignored, a language tag on the fence is dropped, and a reply with no fence at all is taken as the query itself.

## OpenAI

{py:class}`~text2cypher.OpenAIClient` uses the Chat Completions API:

```python
from text2cypher import OpenAIClient

llm = OpenAIClient("your-model")
```

* There is no default model: the right one depends on your account and on how large your schema is.
* `temperature` is `0` by default; `temperature=None` leaves it out, for models that only accept their own.
* `system_role=False` sends one user message, the system text followed by the user text, for a model that has no system role.
* It reads `OPENAI_API_KEY` unless you pass your own `AsyncOpenAI` as `client`, for example for an OpenAI-compatible server:

```python
from openai import AsyncOpenAI
from text2cypher import OpenAIClient

llm = OpenAIClient(
    "a-local-model",
    client=AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
)
```

## Testing your code

{py:class}`~text2cypher.FakeLLM` replays scripted replies in order and records what it was asked in `calls`, so you can test code that uses text2cypher without a model. A reply is a string, or a `Completion` to give token usage:

```python
from text2cypher import FakeLLM, Text2Cypher

llm = FakeLLM(["```cypher\nMATCH (p:Person) RETURN p.name\n```"])
result = await Text2Cypher(client, llm).run("Who is in the graph?", graph="movies")

assert result.status == "valid"
assert llm.calls[0].user == "Question: Who is in the graph?"
```

It raises when it is asked more times than it has replies. See also [](usage.md#try-it-without-a-server-or-a-model).
