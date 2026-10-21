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
* The client is used with the defaults of `OpenAIClient` (below). For its options, make one yourself.

{py:class}`~text2cypher.OpenAIClient` is the same thing with its options. It uses the Chat Completions API:

```python
from text2cypher import OpenAIClient

llm = OpenAIClient("your-model")
```

`OpenAIClient.with_model("another-model")` is a copy that asks another model with the same SDK client and settings.

* There is no default model: the right one depends on your account and on how large your schema is.
* `temperature` is `0` by default; `temperature=None` leaves it out, for models that only accept their own.
* `system_role=False` puts the system text before the text of the first user turn, for a model that has no system role.
* `timeout` (120 seconds by default) is how long a call waits for a reply before it is given up and retried. The OpenAI SDK waits 10 minutes, and a question can make a dozen calls, so a hung connection could hold a run for hours. Raise it for a model that thinks for longer; `timeout=None` leaves the timeout of the client you pass. It applies to your own `AsyncOpenAI` too.
* `tools=False` says the model or server cannot be offered tools (some OpenAI-compatible servers cannot), so the library offers none.
* `max_retries` is how many times the SDK retries a call that fails on a timeout, a connection error, a rate limit or a server error, with a pause between. `None` (the default) leaves the client's own setting, which is 2 for the one made here. When a call still fails, the run ends with an `LLMError` that keeps the attempts made so far (see [Errors](usage.md#errors)).
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

It raises when it is asked more times than it has replies. See also [Try it without a server or a model](usage.md#try-it-without-a-server-or-a-model).
