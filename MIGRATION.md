# Migrating from 2.x to 3.0

3.0 changes one thing: how `OpenAIClient` is given its settings. If you never gave it any (`OpenAIClient("model")`, or the SDK's own client with `model=`), nothing changes.

The keywords `temperature=`, `system_role=`, `timeout=`, `max_retries=` and `tools=` are gone, and they are now keys of one `model_options` dictionary, with two more (`reasoning_effort` and `extra`):

Before (2.x):

```python
OpenAIClient("model", client=sdk, temperature=None, timeout=300, tools=False)
```

After (3.0):

```python
OpenAIClient(
    "model",
    client=sdk,
    model_options={"temperature": None, "timeout": 300, "tools": False},
)
```

* The values and defaults are the same, so a call that sets one is a one-for-one rewrite. Passing an old keyword is a `TypeError`.
* A key that is not a setting is a `ValueError` that names it, and a type checker flags it too.
* The same dictionary now works elsewhere: `Text2Cypher(client, AsyncOpenAI(), model="m", model_options={...})`, `LLMSchemaFilter(model="small", model_options={...})`, and `OpenAIClient.with_model("other", {...})`. A model that reasons wants `{"temperature": None, "reasoning_effort": "minimal"}`.
* **The prompts are Markdown.** Each part has a heading (`# Rules`, `# Schema`, and `# Question` for the question) in place of `Rules:`, `Schema:` and `Question: ...`. A test that compares a prompt (`llm.calls[0].user == "Question: ..."`), or a client of your own that picks the question out of the user message, must follow. The wording of the rules is unchanged, and `render_schema` keeps its text.
* `LLMSchemaFilter(model="small")` with no client of its own used to ask the model with every setting of the main client. It now asks with no temperature (the server's own, which every model accepts; a reasoning model refuses 0), no `reasoning_effort` and no `extra`, and keeps the main client's `timeout`, `max_retries`, `system_role` and `tools`. If you want a temperature for the filter's model, such as 0 for a choice that does not vary, say so in `model_options`.

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

A `Completion` also has `finish_reason`, which a custom client can set (`"stop"`, `"length"`, `"content_filter"`, `"refusal"`) and leave `None` if it does not know. Code that unpacks them into a fixed number of names does not (`text, usage = completion`, or `system, user = call`). Read the fields by name instead: `completion.text`, `completion.usage`, `call.system`, `call.user`.

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
