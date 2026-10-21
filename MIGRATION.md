# Migrating from 1.x to 2.0

2.0 changes one thing you may have written yourself: the `chat` method of an LLM client. Everything else you may use keeps working. This page says what to change, and how to tell whether it affects you.

## A client of your own with a `chat` method

`chat` now takes the tools a model may be offered. Add the two arguments to your method; if your client cannot call tools, ignore them.

Before (1.x):

```python
class MyClient:
    async def chat(self, system: str, messages: Sequence[Message]) -> Completion: ...
```

After (2.0):

```python
from typing import Literal

from text2cypher import Completion, Message, Tool


class MyClient:
    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> Completion: ...
```

* The library offers tools only for a feature that needs them, and only to a client that can take them: one that has a `supports_tools = True` attribute, or whose `chat` takes `tools` or any keyword arguments. A client that cannot use tools can say `supports_tools = False`. So a 1.x `chat` keeps working at run time; it is the type checker that reports the new signature.
* If your client does take tools, a tool is a name, a description and a JSON Schema for its parameters (`Tool`), the same shape as in the OpenAI SDK and MCP. The model's request comes back as `Completion.tool_calls` (`ToolCall(id, name, arguments)`, with the arguments already a dict), and the library answers it with a `Message` whose `role` is `"tool"` and whose `tool_call_id` is the call's id. `OpenAIClient` is the reference for this.

A client with only a `complete(system, user)` method needs no change at all.

## Code that unpacks a `Completion`, a `Message` or a `Call`

These tuples have more fields, with defaults, so building them as before still works:

```python
Completion(text, usage)  # still fine
Message("user", text)  # still fine
```

Code that unpacks them into a fixed number of names does not (`text, usage = completion`, or `system, user = call`). Read the fields by name instead: `completion.text`, `completion.usage`, `call.system`, `call.user`.

## OpenAI

`OpenAIClient("model")` still works and keeps all its options. The simplest way to use OpenAI is now to pass the SDK's own client and the model:

```python
from openai import AsyncOpenAI
from text2cypher import Text2Cypher

agent = Text2Cypher(client, AsyncOpenAI(), model="your-model")
```

This also works for `AsyncAzureOpenAI` and for a client made for an OpenAI-compatible server. `model=` is required with an SDK client and is an error with any other kind. The synchronous `openai.OpenAI` is refused.

## If you are upgrading from 1.1 or earlier

Read the **Changed** list of 1.2.0 in the changelog too: the higher default limits, example values being sent to the LLM provider, `execute=True` ending as `failed` when a query keeps failing on the database, and an exception from your LLM client arriving as an `LLMError`.
