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
    await client.complete("system", "question")

    assert isinstance(client, OpenAIClient)
    assert requests[0]["model"] == "a-model"
    assert requests[0]["messages"][-1] == {"role": "user", "content": "question"}


async def test_a_client_made_for_a_compatible_server_works_the_same_way():
    # the helper's client points at http://test/v1, not at OpenAI
    sdk, requests = openai_client(REPLY)

    await as_client(sdk, "local-model").complete("s", "u")

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

    await as_client(sdk, "my-deployment").complete("s", "u")

    assert requests[0]["model"] == "my-deployment"


def test_an_async_openai_has_no_model_of_its_own():
    sdk, _ = openai_client(REPLY)

    with pytest.raises(ValueError, match="no model of its own"):
        as_client(sdk)


def test_the_synchronous_client_is_refused_because_the_library_is_async():
    with pytest.raises(TypeError, match="async"):
        as_client(openai.OpenAI(api_key="test"))


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
    main = OpenAIClient("big", client=sdk, temperature=0.5, system_role=False)

    cheap = main.with_model("small")
    await cheap.complete("system", "user")
    await main.complete("system", "user")

    assert [r["model"] for r in requests] == ["small", "big"]
    assert [r["temperature"] for r in requests] == [0.5, 0.5]
    # the system role setting is kept too: one user message with the system text in it
    assert all(len(r["messages"]) == 1 for r in requests)
