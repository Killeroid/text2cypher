# Migrating from 2.x to 3.0

3.0 changes three things: `OpenAIClient` is gone (the OpenAI SDK's own client takes its place), the text of the prompts, and the settings of the LLM schema filter's model. If you do none of these things (you do not use `OpenAIClient`, you do not read or compare a prompt, and you do not use `LLMSchemaFilter(model=...)`), nothing changes for you.

## `OpenAIClient` is gone: give the SDK's client and `model_options`

`OpenAIClient` is no longer part of the API. The OpenAI SDK's own client (`AsyncOpenAI`, `AsyncAzureOpenAI`, or one for a compatible server) does its job, with the model in `model=` and what were its keywords, `temperature=`, `system_role=`, `timeout=`, `max_retries=` and `tools=`, now keys of one `model_options` dictionary, with two more (`reasoning_effort` and `extra`):

| 2.x | 3.0 |
|---|---|
| `Text2Cypher(client, OpenAIClient("m"))` | `Text2Cypher(client, AsyncOpenAI(), model="m")` |
| `OpenAIClient("m", client=sdk, temperature=None, timeout=300)` | `Text2Cypher(client, sdk, model="m", model_options={"temperature": None, "timeout": 300})` |
| `LLMSchemaFilter(OpenAIClient("small", client=sdk))` | `LLMSchemaFilter(sdk, model="small")` |
| `OpenAIClient("main").with_model("small")` as the filter's client | `LLMSchemaFilter(model="small")`, which asks the main client with that model |
| `OpenAIClient("m", client=sdk)` used on its own, `await llm.complete(system, user)` | the SDK's own `sdk.chat.completions.create(...)`, or a client of your own with a `chat` method (`AsyncChatClient`) |

* The values and defaults of the settings are the same. Passing an old keyword is a `TypeError`, and `from text2cypher import OpenAIClient` is an `ImportError`.
* A key that is not a setting is a `ValueError` that names it, and a type checker flags it too.
* `model_options` works for the main model and for the filter's model: `Text2Cypher(client, AsyncOpenAI(), model="m", model_options={...})` and `LLMSchemaFilter(model="small", model_options={...})`. A model that reasons wants `{"temperature": None, "reasoning_effort": "minimal"}`.
* A client of your own (one with a `chat` or a `complete` method) is unchanged, and has its own settings: `model_options` is refused with it.

## The prompts are Markdown

Each part of a prompt has a heading (`# Rules`, `# Schema`, and `# Question` for the question) in place of `Rules:`, `Schema:` and `Question: ...`. A test that compares a prompt (`llm.calls[0].user == "Question: ..."`), or a client of your own that picks the question out of the user message, must follow. The wording of the rules is unchanged, and `render_schema` keeps its text.

## The filter's model

`LLMSchemaFilter(model="small")` with no client of its own used to ask the model with every setting of the main client. It now asks with no temperature (the server's own, which every model accepts; a reasoning model refuses 0), no `reasoning_effort` and no `extra`, and keeps the main client's `timeout`, `max_retries`, `system_role` and `tools`. The same holds for an `AsyncOpenAI` given to the filter with `model=`. If you want a temperature for the filter's model, such as 0 for a choice that does not vary, say so in `model_options`.

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

`OpenAIClient("model")` still works and keeps all its options (in 3.0 it is gone: see above). The simplest way to use OpenAI is now to pass the SDK's own client and the model:

```python
from openai import AsyncOpenAI
from text2cypher import Text2Cypher

agent = Text2Cypher(client, AsyncOpenAI(), model="your-model")
```

This also works for `AsyncAzureOpenAI` and for a client made for an OpenAI-compatible server. `model=` is required with an SDK client and is an error with any other kind. The synchronous `openai.OpenAI` is refused.

## If you are upgrading from 1.1 or earlier

Read the **Changed** list of 1.2.0 in the changelog too: the higher default limits, example values being sent to the LLM provider, `execute=True` ending as `failed` when a query keeps failing on the database, and an exception from your LLM client arriving as an `LLMError`.
