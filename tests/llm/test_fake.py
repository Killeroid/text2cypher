import pytest

from text2cypher.llm import Completion, FakeLLM, Usage

pytestmark = pytest.mark.anyio


async def test_fake_llm_replies_in_order_and_records_the_calls():
    llm = FakeLLM(["first", Completion("second", Usage(3, 4))])

    first = await llm.complete("system 1", "user 1")
    second = await llm.complete("system 2", "user 2")

    assert first == Completion("first", None)
    assert second == Completion("second", Usage(3, 4))
    assert [(call.system, call.user) for call in llm.calls] == [
        ("system 1", "user 1"),
        ("system 2", "user 2"),
    ]


async def test_fake_llm_fails_when_it_runs_out_of_replies():
    llm = FakeLLM(["only"])
    await llm.complete("s", "u")

    with pytest.raises(RuntimeError, match="no reply left"):
        await llm.complete("s", "u")
