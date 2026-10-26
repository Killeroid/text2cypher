# LLM clients

## The protocol

An LLM is anything with a `chat` method ({py:class}`~text2cypher.AsyncChatClient`) or a `complete` method ({py:class}`~text2cypher.AsyncLLMClient`). Either returns the text and, if it knows them, the tokens used.

`chat` is the one text2cypher calls when a client has it. It gets a system message and the turns of the conversation so far, each a {py:class}`~text2cypher.Message`: the question, then, for every query the model wrote that had to be corrected, that query as an `assistant` turn and a `user` turn that says what was wrong with it. The first and the last turn are always from the user.

```python
from collections.abc import Sequence

from text2cypher import Completion, Message, Text2Cypher, Usage


class MyClient:
    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools=None,
        *,
        tool_choice="auto",
    ) -> Completion:
        text, tokens_in, tokens_out = await ask_my_model(system, messages)
        return Completion(text, Usage(input_tokens=tokens_in, output_tokens=tokens_out))


agent = Text2Cypher(client, MyClient())
```

`tools` and `tool_choice` are for a model that can call functions, and a client that cannot may ignore them: it says so with `supports_tools = False`, and the library then offers it none. A {py:class}`~text2cypher.Tool` is a name, a description and a JSON Schema for its parameters, the shape the OpenAI SDK and MCP use. When a model asks for one, the reply has `tool_calls` ({py:class}`~text2cypher.ToolCall`: an id, a name and the arguments as a dict), and the library answers with a `Message` whose `role` is `"tool"` and whose `tool_call_id` is the call's id, then asks again. `tool_choice="none"` offers the tools but forbids a call, to make the model answer. From 1.x, see [Migrating from 1.x to 2.0](migration.md). The library offers tools only for `Config(use_skills=True)`: one tool, `read_skill`, with one parameter, `id`. A client that does not take tools is given the skills in the prompt instead, so it still works.

A client with only `complete(system, user)` still works, and every client written for it before `chat` existed keeps working. It gets the same conversation as one user message: the question, then the earlier attempts, oldest first, each with its problems, and then the instruction to correct them.

```python
class MyClient:
    async def complete(self, system: str, user: str) -> Completion:
        text, tokens_in, tokens_out = await ask_my_model(system, user)
        return Completion(text, Usage(input_tokens=tokens_in, output_tokens=tokens_out))
```

text2cypher keeps no state between questions, and the models are stateless: every call carries everything the model needs, the system message and the turns. Use `Completion(text, None)` when the provider does not report tokens; `result.usage` is then `None`. If you know why a reply ended, say so with `Completion(text, usage, finish_reason=...)`: `"stop"`, `"length"` (a token limit), `"content_filter"` (the provider blocked it) or `"refusal"` (the model refused). It matters only for a reply with no text, where it decides whether to ask again (see [How it works](how-it-works.md#when-the-llm-sends-nothing)); leave it out and an empty reply is asked again once. If your client has both methods, `chat` is used; an attribute named `chat` that is not callable does not count.

The reply should hold the query in one fenced block (` ```cypher ... ``` `). An empty block means the question cannot be answered. Text around the block is ignored, a language tag on the fence is dropped, and a reply with no fence at all is taken as the query itself.

## OpenAI

The simplest way is to pass the OpenAI SDK's own client and say which model:

```python
from openai import AsyncOpenAI
from text2cypher import Text2Cypher

agent = Text2Cypher(client, AsyncOpenAI(), model="your-model")
```

* Any `openai.AsyncOpenAI` works, which includes `AsyncAzureOpenAI` (the model is then the deployment name) and a client made for an OpenAI-compatible server with `base_url`. Your own configuration of it (key, headers, proxies, retries) is used as it is.
* It has no model of its own, so `model=` is required, and giving `model=` with any other kind of client is an error.
* The synchronous `openai.OpenAI` is refused, because the library is async.
* The client is asked with the defaults of the model options (below). To ask it differently, add `model_options`.
* `model_options` is refused with any other kind of client, which has its own settings, so it is never silently ignored.

```python
agent = Text2Cypher(
    client,
    AsyncOpenAI(),
    model="your-reasoning-model",
    model_options={"temperature": None, "reasoning_effort": "minimal"},
)
```

For an OpenAI-compatible server, give your own `AsyncOpenAI`:

```python
agent = Text2Cypher(
    client,
    AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed"),
    model="a-local-model",
)
```

The LLM schema filter asks a model of its own the same way: `LLMSchemaFilter(model="small", model_options={...})` asks the main client with another model, and `LLMSchemaFilter(AsyncOpenAI(), model="small", model_options={...})` asks another client. With no client of its own the filter's model starts with no `temperature` (the server's own: every model takes that, and a reasoning model refuses 0, which the filter would hide by keeping the whole schema, so the choice is less repeatable; `{"temperature": 0}` makes it so), no `reasoning_effort` and no `extra`, whatever the main client has, because those depend on the model; it keeps the main client's `timeout`, `max_retries`, `system_role` and `tools`, which belong to the server. With an `AsyncOpenAI` as its client the filter starts the same way. A model that does not reason can have `{"reasoning_effort": None}` and its own `temperature`.

### Model options

`model_options` is a {py:class}`~text2cypher.ModelOptions` dictionary. Every key is optional, and a key that is not one of these is an error that names it (a type checker shows it too).

* There is no default model: the right one depends on your account and on how large your schema is.
* `temperature` is `0` by default; `None` leaves it out, for models that only accept their own (reasoning models). For Claude it cannot be set (see [Claude](#claude-anthropic)).
* `reasoning_effort` (`"minimal"`, `"low"`, `"medium"`, `"high"`, or what the server takes) is how much a reasoning model thinks; `None` (the default) leaves it out, so a request is the same as without it.
* `extra` is a mapping of other request fields sent on every call, such as `max_completion_tokens`, `seed` or `service_tier`; a name the SDK does not know is sent as it is. `model`, `messages`, `temperature`, `tools`, `tool_choice`, `reasoning_effort` and `stream` are set by the client, so `extra` naming one is a `ValueError`. A new `extra` replaces the old one as a whole.
* `system_role=False` puts the system text before the text of the first user turn, for a model that has no system role.
* `timeout` (120 seconds by default) is how long a call waits for a reply before it is given up and retried. The OpenAI SDK waits 10 minutes, and a question can make a dozen calls, so a hung connection could hold a run for hours. Raise it for a model that thinks for longer; `None` leaves the timeout of the client you pass. It applies to your own `AsyncOpenAI` too.
* `tools=False` says the model or server cannot be offered tools (some OpenAI-compatible servers cannot), so the library offers none.
* `max_retries` is how many times the SDK retries a call that fails on a timeout, a connection error, a rate limit or a server error, with a pause between. `None` (the default) leaves the setting of the client you pass, which is 2 for a default `AsyncOpenAI`. When a call still fails, the run ends with an `LLMError` that keeps the attempts made so far (see [Errors](usage.md#errors)).
* The SDK reads `OPENAI_API_KEY` for `AsyncOpenAI()`, or you pass your own key, `base_url` and other configuration to it.

### Model Selection

Choose llm models based on your requirements. Different models provide different balances of speed and accuracy and cost. See an example comparison:

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| gpt-4o-mini | Fast | Good | Low | Development, simple queries |
| gpt-4o | Medium | Better | Medium | Production, complex queries |
| gpt-4 | Slow | Best | High | Critical queries, complex schemas |


## Claude (Anthropic)

Install the extra, then pass the Anthropic SDK's own client and say which model, as with OpenAI:

```sh
uv add "text2cypher[anthropic]"        # or: pip install "text2cypher[anthropic]"
```

```python
from anthropic import AsyncAnthropic
from text2cypher import Text2Cypher

agent = Text2Cypher(client, AsyncAnthropic(), model="claude-sonnet-5-5")
```

* Any `anthropic.AsyncAnthropic` works; the SDK reads `ANTHROPIC_API_KEY`, or you pass your own key and configuration to it. It has no model of its own, so `model=` is required, and `model=` and `model_options` are refused with a client that is not an OpenAI or Anthropic SDK client, as with OpenAI.
* `LLMSchemaFilter(model="claude-haiku-5-5")` asks the main client with another model, and `LLMSchemaFilter(AsyncAnthropic(), model="...")` asks another client, as with OpenAI.
* **There is no `temperature`.** The current models take only their default (and on the newest API version, which the SDK sends from 2.0.0, none of `temperature`, `top_p` and `top_k` at all, even at the default), so the adapter never sends one and `model_options={"temperature": ...}` is a `ValueError`. A Claude model's reply cannot be made repeatable the way `temperature: 0` does for another model.
* `reasoning_effort` (`"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`) is sent as the effort of the request. Leave it out to take the model's own default, which differs by model. The models think by default, and some cannot be told not to.
* `max_tokens` is required by the API, and the model's thinking counts toward it: the default is 16000, and `model_options={"extra": {"max_tokens": 32000}}` changes it. The rest of `extra` is sent as it is; it may not name `system`, `top_p` or `top_k`.
* `timeout`, `max_retries` and `tools` work as with OpenAI (see [Model options](#model-options)). `system_role` has no effect: the system prompt is always its own field.
* A reply that the model refuses to give ends the call with the finish reason `"refusal"`, which the loop does not retry.
* **Skills.** When a reply asks for a tool (a skill is read with one), the API requires the model's thinking blocks to come back unchanged with the tool's answer. The adapter keeps the content blocks of the last 256 such replies and sends them back; a reply with no tool call is sent back as its text alone, which the API allows.

## Testing your code

{py:class}`~text2cypher.FakeLLM` replays scripted replies in order and records what it was asked in `calls`, so you can test code that uses text2cypher without a model. A reply is a string, or a `Completion` to give token usage:

```python
from text2cypher import FakeLLM, Text2Cypher

llm = FakeLLM(["```cypher\nMATCH (p:Person) RETURN p.name\n```"])
result = await Text2Cypher(client, llm).run("Who is in the graph?", graph="movies")

assert result.status == "valid"
assert llm.calls[0].user == "# Question\nWho is in the graph?"
```

It raises when it is asked more times than it has replies. See also [Try it without a server or a model](usage.md#try-it-without-a-server-or-a-model).
