import asyncio
import pickle

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from support import FILM_SCHEMA, ScriptedValidator, invalid, reply

from text2cypher import (
    Completion,
    Config,
    DatabaseError,
    FakeLLM,
    GraphNotFoundError,
    LLMError,
    Text2Cypher,
    Text2CypherError,
    Usage,
)
from text2cypher._results import Attempt
from text2cypher._workflow import run_workflow

pytestmark = pytest.mark.anyio


class RateLimited(Exception):
    pass


async def test_the_llm_failing_on_the_first_call_ends_the_run_with_nothing_to_keep():
    cause = RateLimited("slow down")

    with pytest.raises(LLMError) as raised:
        await run_workflow(
            "Who?",
            FILM_SCHEMA,
            llm=FakeLLM([cause]),
            validate=ScriptedValidator(),
            config=Config(),
        )

    error = raised.value
    assert error.__cause__ is cause
    assert error.question == "Who?"
    assert error.attempts == ()
    assert error.usage == Usage(0, 0)
    assert str(error) == "The LLM failed after 0 attempt(s) at 'Who?': slow down"


async def test_the_llm_failing_on_a_correction_keeps_the_attempts_and_what_they_cost():
    first = Completion(reply("MATCH (x:X) RETURN x"), Usage(10, 4))
    second = Completion(reply("MATCH (y:Y) RETURN y"), Usage(12, 5))
    cause = RateLimited("slow down")

    with pytest.raises(LLMError) as raised:
        await run_workflow(
            "Who?",
            FILM_SCHEMA,
            llm=FakeLLM([first, second, cause]),
            validate=ScriptedValidator(invalid("x"), invalid("y")),
            config=Config(),
        )

    error = raised.value
    assert error.__cause__ is cause
    assert [a.query for a in error.attempts] == [
        "MATCH (x:X) RETURN x",
        "MATCH (y:Y) RETURN y",
    ]
    assert error.attempts[0].validation == invalid("x")
    # the call that failed is not counted: nothing says what it cost
    assert error.usage == Usage(22, 9)
    assert "after 2 attempt(s)" in str(error)


async def test_the_usage_is_unknown_when_an_attempt_did_not_report_it():
    with pytest.raises(LLMError) as raised:
        await run_workflow(
            "Q",
            FILM_SCHEMA,
            llm=FakeLLM([reply("a"), RateLimited("x")]),
            validate=ScriptedValidator(invalid("x")),
            config=Config(),
        )

    assert raised.value.usage is None


async def test_cancelling_the_task_is_not_an_llm_error():
    class Cancelled:
        async def complete(self, system: str, user: str) -> Completion:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_workflow(
            "Q",
            FILM_SCHEMA,
            llm=Cancelled(),
            validate=ScriptedValidator(),
            config=Config(),
        )


async def test_a_database_failure_while_validating_keeps_the_query_it_was_validating():
    class LostConnection:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, query: str):
            self.calls += 1
            if self.calls == 1:
                return invalid("x")
            raise RedisConnectionError("connection lost")

    cause_text = "connection lost"
    llm = FakeLLM(
        [
            Completion(reply("MATCH (x:X) RETURN x"), Usage(10, 4)),
            Completion(reply("MATCH (p:Person) RETURN p"), Usage(12, 5)),
        ]
    )

    with pytest.raises(DatabaseError) as raised:
        await run_workflow(
            "Who?",
            FILM_SCHEMA,
            llm=llm,
            validate=LostConnection(),
            config=Config(),
        )

    error = raised.value
    assert isinstance(error.__cause__, RedisConnectionError)
    assert str(error) == (
        f"The database failed after 2 attempt(s) at 'Who?': {cause_text}"
    )
    first, unchecked = error.attempts
    assert first.validation == invalid("x")
    # the second query was written and paid for, and has no validation
    assert unchecked == Attempt("MATCH (p:Person) RETURN p", None, "full", Usage(12, 5))
    assert error.usage == Usage(22, 9)


async def test_an_error_in_validation_that_is_not_the_databases_is_a_bug_and_propagates():
    class Broken:
        async def __call__(self, query: str):
            raise TypeError("a bug")

    with pytest.raises(TypeError, match="a bug"):
        await run_workflow(
            "Q",
            FILM_SCHEMA,
            llm=FakeLLM([reply("a")]),
            validate=Broken(),
            config=Config(),
        )


def test_the_errors_have_one_base_and_the_graph_error_is_still_a_value_error():
    assert issubclass(LLMError, Text2CypherError)
    assert issubclass(DatabaseError, Text2CypherError)
    assert issubclass(GraphNotFoundError, Text2CypherError)
    assert issubclass(GraphNotFoundError, ValueError)


def test_an_llm_error_survives_being_pickled_with_what_it_carries():
    attempts = (Attempt("MATCH (n) RETURN n", None, "full", Usage(3, 2)),)
    error = LLMError.after("Q", attempts, RateLimited("x"))

    again = pickle.loads(pickle.dumps(error))

    assert isinstance(again, LLMError)
    assert (again.question, again.attempts, again.usage) == ("Q", attempts, Usage(3, 2))
    assert str(again) == str(error)


async def test_a_run_on_a_real_graph_that_loses_the_llm_keeps_the_first_attempt(
    falkordb_client, movies_graph: str
):
    llm = FakeLLM(
        [
            Completion(reply("MATCH (x:Nothing) RETURN x"), Usage(7, 3)),
            RateLimited("slow down"),
        ]
    )

    with pytest.raises(LLMError) as raised:
        await Text2Cypher(falkordb_client, llm).run("Who?", movies_graph)

    error = raised.value
    assert [a.query for a in error.attempts] == ["MATCH (x:Nothing) RETURN x"]
    assert error.attempts[0].validation is not None
    assert not error.attempts[0].validation.valid
    assert error.usage == Usage(7, 3)
    assert isinstance(error.__cause__, RateLimited)
