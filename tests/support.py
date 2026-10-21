"""Helpers shared by the tests: builders for the values the library passes around."""

import json
from typing import Any, cast

import httpx2 as httpx
from cypher_guardrail import issues as cg
from openai import AsyncOpenAI

from text2cypher import GraphSchema, Issue, Records, ValidationResult
from text2cypher._graph.execution import Execution
from text2cypher.schema import Path

# A small schema, built by hand: (:Person {name})-[:ACTED_IN]->(:Movie {title})
FILM_SCHEMA = GraphSchema(
    node_properties={"Movie": {"title": ("String",)}, "Person": {"name": ("String",)}},
    relationship_properties={"ACTED_IN": {}},
    paths=(Path("Person", "ACTED_IN", "Movie"),),
)

VALID = ValidationResult((), True, 1.0, 1.0)


def reply(query: str) -> str:
    """What an LLM answers with: the query in a fenced block."""
    return f"```cypher\n{query}\n```"


def make_issue(
    code: str, description: str = "", source: str = "schema", **fields: Any
) -> Issue:
    """An issue with the given code and fields, as a validator would report it."""
    raw = {"code": code, "description": description, **fields}
    return Issue(source=cast("Any", source), raw=cast("Any", raw))


def invalid(description: str, code: str = cg.UNKNOWN_LABEL) -> ValidationResult:
    """A validation result with one issue (an unknown label gets the field it always has)."""
    fields = {"label": "Robot"} if code == cg.UNKNOWN_LABEL else {}
    return ValidationResult((make_issue(code, description, **fields),), True, 0.0, None)


def misspelled(
    *, label: str | None = None, type: str | None = None, prop: str | None = None
) -> ValidationResult:
    """A validation result for a query that uses a name the graph does not have."""
    issues = []
    if label:
        issues.append(make_issue(cg.UNKNOWN_LABEL, "No label.", label=label))
    if type:
        issues.append(
            make_issue(cg.UNKNOWN_REL_TYPE, "No type.", relationship_type=type)
        )
    if prop:
        issues.append(make_issue(cg.UNKNOWN_PROPERTY, "No property.", property=prop))
    return ValidationResult(tuple(issues), True, 0.0, None)


class ScriptedValidator:
    """Answers each call with the next canned result, and records the queries."""

    def __init__(self, *results: ValidationResult) -> None:
        self._results = iter(results)
        self.queries: list[str] = []

    async def __call__(self, query: str) -> ValidationResult:
        self.queries.append(query)
        return next(self._results)


ROWS = Records(("x",), ((1,),))
RAN = Execution(records=ROWS)


def failed_to_run(error: str) -> Execution:
    """FalkorDB failed the query: another query may not fail."""
    return Execution(error=error, correctable=True)


class ScriptedExecutor:
    """Answers each run with the next canned execution, and records the queries."""

    def __init__(self, *executions: Execution) -> None:
        self._executions = iter(executions)
        self.queries: list[str] = []

    async def __call__(self, query: str) -> Execution:
        self.queries.append(query)
        return next(self._executions)


def openai_client(
    reply: dict,
    requests: list[dict] | None = None,
    *,
    failures: int = 0,
    timeouts: list[float | None] | None = None,
    replies: list[dict] | None = None,
    **options: Any,
) -> tuple[AsyncOpenAI, list[dict]]:
    """A real ``AsyncOpenAI`` whose HTTP layer answers with ``reply``, after ``failures`` 500s.

    The read timeout each request was sent with is added to ``timeouts``. With ``replies``,
    the requests are answered with them in turn, and with ``reply`` after they run out.
    """
    seen: list[dict] = [] if requests is None else requests
    failed = 0
    queue = iter(replies or [])

    def answer(request: httpx.Request) -> httpx.Response:
        nonlocal failed
        seen.append(json.loads(request.content))
        if timeouts is not None:
            timeouts.append(request.extensions["timeout"]["read"])
        if failed < failures:
            failed += 1
            return httpx.Response(500, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=next(queue, reply))

    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)),
        **options,
    )
    return client, seen


def chat_reply(content: str | None, usage: dict | None) -> dict:
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if usage is not None:
        reply["usage"] = usage
    return reply


def tool_calls_reply(
    calls: list[tuple[str, str, str]],
    usage: dict | None = None,
    *,
    kind: str = "function",
) -> dict:
    """A chat reply in which the model asks for tools: ``(id, name, arguments as JSON)`` each."""
    reply = chat_reply(None, usage)
    reply["choices"][0]["finish_reason"] = "tool_calls"
    reply["choices"][0]["message"]["tool_calls"] = [
        {"id": id_, "type": kind, "function": {"name": name, "arguments": arguments}}
        for id_, name, arguments in calls
    ]
    return reply


def chat_reply_ending(
    content: str | None, finish_reason: str, *, refusal: str | None = None
) -> dict:
    """A chat reply that ended for ``finish_reason``, perhaps with a refusal and no content."""
    reply = chat_reply(content, None)
    reply["choices"][0]["finish_reason"] = finish_reason
    if refusal is not None:
        reply["choices"][0]["message"]["refusal"] = refusal
    return reply
