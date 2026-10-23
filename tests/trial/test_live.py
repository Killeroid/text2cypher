"""The settings and the clients of the measurement, without any network or any real key."""

import pytest
from support import chat_reply, openai_client

from evals.live import REQUIRED, load_env, make_client, parse_env
from text2cypher.llm._chat_completions import ChatCompletionsAdapter

pytestmark = pytest.mark.anyio

REPLY = chat_reply(
    "```cypher\nRETURN 1\n```", {"prompt_tokens": 5, "completion_tokens": 2}
)


def test_settings_can_be_written_with_a_colon_or_an_equals_sign():
    text = (
        "# a comment\n"
        "\n"
        "OPENAI_API_KEY: sk-not-real\n"
        "OPENAI_BASE_URL: https://example.test/openai/v1\n"
        "CHEAP_MODEL=small\n"
        "export EXPENSIVE_MODEL = 'big'\n"
        'EXPENSIVE_MODEL_REASONING_EFFORT: "minimal"\n'
    )

    assert (
        parse_env(text)
        == {
            "OPENAI_API_KEY": "sk-not-real",
            "OPENAI_BASE_URL": "https://example.test/openai/v1",  # the colon of the URL is kept
            "CHEAP_MODEL": "small",
            "EXPENSIVE_MODEL": "big",
            "EXPENSIVE_MODEL_REASONING_EFFORT": "minimal",
        }
    )


def test_a_commented_out_setting_is_not_one():
    assert parse_env("#CHEAP_MODEL: small\n  # EXPENSIVE_MODEL: big") == {}


def test_load_env_puts_the_settings_in_the_environment_and_the_environment_wins(
    tmp_path, monkeypatch
):
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{name}: from-file" for name in REQUIRED))
    monkeypatch.setenv("CHEAP_MODEL", "from-environment")

    load_env(path)

    import os

    assert os.environ["CHEAP_MODEL"] == "from-environment"
    assert os.environ["EXPENSIVE_MODEL"] == "from-file"


def test_a_missing_setting_is_named_and_no_value_is_in_the_error(tmp_path, monkeypatch):
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY: sk-this-must-not-be-printed\n")

    with pytest.raises(RuntimeError) as error:
        load_env(path)

    assert "CHEAP_MODEL" in str(error.value)
    assert "OPENAI_API_KEY" not in str(error.value)  # it is there
    assert "sk-this-must-not-be-printed" not in str(error.value)


async def test_the_expensive_model_is_asked_with_its_reasoning_effort_and_no_temperature(
    monkeypatch: pytest.MonkeyPatch,
):
    sdk, requests = openai_client(REPLY)
    for name, value in {
        "OPENAI_API_KEY": "k",
        "OPENAI_BASE_URL": "http://test/v1",
        "CHEAP_MODEL": "small",
        "EXPENSIVE_MODEL": "big",
        "EXPENSIVE_MODEL_REASONING_EFFORT": "minimal",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("evals.live.AsyncOpenAI", lambda: sdk)

    await make_client("expensive").complete("system", "question")
    await make_client("cheap").complete("system", "question")

    assert requests[0]["reasoning_effort"] == "minimal"
    assert "temperature" not in requests[0]
    assert requests[0]["model"] == "big"
    assert requests[1]["temperature"] == 0
    assert "reasoning_effort" not in requests[1]


async def test_the_cheap_model_is_asked_with_temperature_zero_and_no_effort():
    sdk, requests = openai_client(REPLY)

    await ChatCompletionsAdapter(
        "small", client=sdk, model_options={"temperature": 0}
    ).complete("system", "question")

    assert requests[0]["temperature"] == 0
    assert "reasoning_effort" not in requests[0]
