import json

import httpx2 as httpx
import openai
import pytest
from support import chat_reply, openai_client

from text2cypher import FakeLLM, OpenAIClient
from text2cypher.llm._adapt import as_client

pytestmark = pytest.mark.anyio

REPLY = chat_reply("```cypher\nRETURN 1\n```", None)


async def test_an_async_openai_with_a_model_is_asked_that_model():
    sdk, requests = openai_client(REPLY)

    client = as_client(sdk, "a-model")
    assert isinstance(client, OpenAIClient)
    await client.complete("system", "question")

    assert requests[0]["model"] == "a-model"
    assert requests[0]["messages"][-1] == {"role": "user", "content": "question"}


async def test_a_client_made_for_a_compatible_server_works_the_same_way():
    # the helper's client points at http://test/v1, not at OpenAI
    sdk, requests = openai_client(REPLY)

    client = as_client(sdk, "local-model")
    assert isinstance(client, OpenAIClient)
    await client.complete("s", "u")

    assert requests[0]["model"] == "local-model"


async def test_an_azure_client_is_an_async_openai_and_the_model_is_the_deployment():
    requests: list[dict] = []

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=REPLY)

    sdk = openai.AsyncAzureOpenAI(
        api_key="test",
        azure_endpoint="http://test",
        api_version="2024-06-01",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
    )

    client = as_client(sdk, "my-deployment")
    assert isinstance(client, OpenAIClient)
    await client.complete("s", "u")

    assert requests[0]["model"] == "my-deployment"


def test_an_async_openai_has_no_model_of_its_own():
    sdk, _ = openai_client(REPLY)

    with pytest.raises(ValueError, match="no model of its own"):
        as_client(sdk)


def test_the_synchronous_client_is_refused_because_the_library_is_async():
    with pytest.raises(TypeError, match="async"):
        as_client(openai.OpenAI(api_key="test"))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("kind", ["fake", "openai"])
def test_a_model_with_a_client_that_is_not_an_async_openai_is_an_error(kind: str):
    sdk, _ = openai_client(REPLY)
    llm = FakeLLM([]) if kind == "fake" else OpenAIClient("m", client=sdk)

    with pytest.raises(ValueError, match=r"is for an openai\.AsyncOpenAI"):
        as_client(llm, "a-model")


def test_any_other_client_is_used_as_it_is():
    llm = FakeLLM([])

    assert as_client(llm) is llm


async def test_with_model_asks_another_model_with_the_same_settings():
    sdk, requests = openai_client(REPLY)
    main = OpenAIClient(
        "big",
        client=sdk,
        model_options={"temperature": 0.5, "system_role": False},
    )

    cheap = main.with_model("small")
    await cheap.complete("system", "user")
    await main.complete("system", "user")

    assert [r["model"] for r in requests] == ["small", "big"]
    assert [r["temperature"] for r in requests] == [0.5, 0.5]
    # the system role setting is kept too: one user message with the system text in it
    assert all(len(r["messages"]) == 1 for r in requests)


async def test_reasoning_effort_and_extra_fields_are_sent_on_every_call():
    sdk, requests = openai_client(REPLY)
    client = OpenAIClient(
        "m",
        client=sdk,
        model_options={
            "reasoning_effort": "minimal",
            "extra": {"max_completion_tokens": 50, "vendor_flag": True},
        },
    )

    await client.complete("s", "u")

    assert requests[0]["reasoning_effort"] == "minimal"
    assert requests[0]["max_completion_tokens"] == 50
    assert requests[0]["vendor_flag"] is True


async def test_without_them_the_request_has_neither():
    sdk, requests = openai_client(REPLY)

    await OpenAIClient("m", client=sdk).complete("s", "u")

    assert set(requests[0]) == {"model", "messages", "temperature"}


@pytest.mark.parametrize("name", ["model", "temperature", "reasoning_effort", "tools"])
def test_extra_cannot_set_what_the_client_sets(name: str):
    sdk, _ = openai_client(REPLY)

    with pytest.raises(ValueError, match=name):
        OpenAIClient("m", client=sdk, model_options={"extra": {name: 1}})


def test_a_setting_that_does_not_exist_is_refused_by_name():
    sdk, _ = openai_client(REPLY)

    with pytest.raises(ValueError, match="reasoning_effot"):
        OpenAIClient("m", client=sdk, model_options={"reasoning_effot": "low"})  # ty: ignore[invalid-key, invalid-argument-type]


async def test_with_model_keeps_what_is_not_named_and_clears_with_none():
    sdk, requests = openai_client(REPLY)
    main = OpenAIClient(
        "big",
        client=sdk,
        model_options={"reasoning_effort": "low", "extra": {"seed": 1}},
    )

    await main.with_model("small").complete("s", "u")
    await main.with_model("small", {"reasoning_effort": None}).complete("s", "u")
    await main.with_model("small", {"extra": {"seed": 2}}).complete("s", "u")
    await main.with_model("small", {"extra": None, "temperature": None}).complete(
        "s", "u"
    )
    await main.complete("s", "u")

    assert [r.get("reasoning_effort") for r in requests] == [
        "low",
        None,
        "low",
        "low",
        "low",
    ]
    assert [r.get("seed") for r in requests] == [1, 1, 2, None, 1]
    assert [r.get("temperature") for r in requests] == [0, 0, 0, None, 0]


async def test_with_model_can_change_the_timeout_and_the_tools_flag():
    timeouts: list[float | None] = []
    sdk, _ = openai_client(REPLY, timeouts=timeouts)
    main = OpenAIClient("big", client=sdk)

    other = main.with_model("small", {"timeout": 7, "tools": False})
    await other.complete("s", "u")

    assert timeouts == [7]
    assert (main.supports_tools, other.supports_tools) == (True, False)


def test_with_model_refuses_an_extra_that_sets_what_the_client_sets():
    sdk, _ = openai_client(REPLY)

    with pytest.raises(ValueError, match="model"):
        OpenAIClient("big", client=sdk).with_model("small", {"extra": {"model": "x"}})


async def test_model_options_on_an_async_openai_are_the_settings_of_the_client_made():
    sdk, requests = openai_client(REPLY)

    client = as_client(sdk, "m", {"temperature": None, "reasoning_effort": "minimal"})
    assert isinstance(client, OpenAIClient)
    await client.complete("s", "u")

    assert requests[0]["reasoning_effort"] == "minimal"
    assert "temperature" not in requests[0]


@pytest.mark.parametrize("kind", ["fake", "openai"])
def test_model_options_with_a_client_that_is_not_an_async_openai_are_refused(kind: str):
    sdk, _ = openai_client(REPLY)
    llm = FakeLLM([]) if kind == "fake" else OpenAIClient("m", client=sdk)

    with pytest.raises(
        ValueError, match=r"model_options is for an openai\.AsyncOpenAI"
    ):
        as_client(llm, None, {"temperature": 1})


def test_the_settings_of_model_options_are_the_ones_of_the_defaults():
    # a setting added to the TypedDict and not to the defaults (or the other way) is caught
    from text2cypher.llm._openai import _DEFAULTS, ModelOptions

    assert set(ModelOptions.__annotations__) == set(_DEFAULTS)


async def test_a_change_to_the_extra_given_afterwards_does_not_change_the_client():
    sdk, requests = openai_client(REPLY)
    extra = {"seed": 1}
    client = OpenAIClient("m", client=sdk, model_options={"extra": extra})

    extra["seed"] = 2
    await client.complete("s", "u")

    assert requests[0]["seed"] == 1
