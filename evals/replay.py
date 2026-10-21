"""Run questions through Text2Cypher with replies that someone else wrote."""

import inspect
import json
import re
import tempfile
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from cypher_guardrail import Snapshot
from falkordb.asyncio import FalkorDB
from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB

from evals.dataset import Dataset, Question, Seeded
from evals.datasets.movies import MOVIES
from evals.judge import filter_note, judge
from text2cypher import (
    Completion,
    Config,
    FilterResult,
    LLMSchemaFilter,
    Message,
    Result,
    Text2Cypher,
    Tool,
    ToolCall,
)
from text2cypher.schema import GraphSchema, filter_schema, schema_from_snapshot

# One correction with the filtered schema, so the escalation can happen
CONFIG = Config(attempts_w_filtered=1, total_attempts=3, execute=True)


class NeedReply(BaseException):
    """The LLM was asked something no reply has been written for yet.

    It is a ``BaseException`` so that nothing between the pipeline and the harness can
    swallow it: the LLM schema filter, for one, keeps the whole schema when its call raises an
    ``Exception``, and would hide that its prompt was never answered.
    """

    def __init__(self, system: str, user: str) -> None:
        self.system, self.user = system, user


class Condition(NamedTuple):
    """What a trial changes about the pipeline: the schema filter and the skills.

    ``filter`` is ``"lexical"`` (the default), ``"llm"`` (an ``LLMSchemaFilter`` that is
    answered like any other call, so its prompt comes first) or ``"none"`` (the full schema).
    """

    filter: Literal["lexical", "llm", "none"] = "lexical"
    skills: bool = False
    example_values: int = 3


_CALL = re.compile(r"^\s*CALL\s+(\w+)\s+(\{.*\})\s*$")


class ReplayLLM:
    """Answers with the replies that were written, in order.

    A reply that is only lines of ``CALL <tool> <JSON arguments>`` is the model calling those
    tools, which is how a model that cannot make a native tool call stands in for one.
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._calls = 0

    async def complete(self, system: str, user: str) -> Completion:
        return self._next(system, user)

    async def chat(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        *,
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> Completion:
        return self._next(system, _transcript(messages, tools, tool_choice))

    def _next(self, system: str, user: str) -> Completion:
        if self._calls == len(self._replies):
            raise NeedReply(system, user)
        self._calls += 1
        return _completion(self._replies[self._calls - 1], self._calls)


def _completion(reply: str, number: int) -> Completion:
    lines = [line for line in reply.strip().splitlines() if line.strip()]
    matches = [_CALL.match(line) for line in lines]
    if lines and all(matches):
        calls = []
        for index, found in enumerate(matches):
            assert found is not None
            try:
                arguments = json.loads(found[2])
            except ValueError:
                arguments = {}
            calls.append(ToolCall(f"call_{number}_{index}", found[1], arguments))
        return Completion("", None, tuple(calls))
    return Completion(reply, None)


def _transcript(
    messages: Sequence[Message], tools: Sequence[Tool] | None, tool_choice: str
) -> str:
    """The conversation as text for whoever answers: the turns, then the tools it may call."""
    if len(messages) == 1 and not tools:
        return messages[0].content
    turns = []
    for message in messages:
        if message.role == "user":
            turns.append(f"USER:\n{message.content}")
        elif message.role == "assistant" and message.tool_calls:
            calls = "\n".join(
                f"CALL {call.name} {json.dumps(dict(call.arguments))}"
                for call in message.tool_calls
            )
            turns.append(f"YOU (calling tools):\n{calls}")
        elif message.role == "assistant":
            turns.append(f"YOU:\n{message.content}")
        else:
            turns.append(
                f"RESULT OF THE CALL {message.tool_call_id}:\n{message.content}"
            )
    text = "\n\n".join(turns)
    if not tools:
        return text
    if tool_choice == "none":
        return text + "\n\nYou may not call a tool now: write the query."
    listing = "\n".join(
        f"- {tool.name}: {tool.description} Arguments, as JSON: {json.dumps(dict(tool.parameters))}"
        for tool in tools
    )
    return (
        f"{text}\n\nYou may call a tool instead of writing the query. To call one, reply with "
        "one line per call and nothing else, exactly: CALL <tool name> <arguments as JSON>. "
        f"The result comes back as the next message. The tools:\n{listing}"
    )


@dataclass(frozen=True, slots=True)
class Pending:
    """A question that needs the reply number ``number``."""

    id: str
    category: str
    number: int
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Finished:
    question: Question | Seeded
    result: Result
    verdict: str
    filter_note: str
    # How many replies the question took, the calls of the filter and of reading skills included
    replies: int = 0


async def load_graph(client: FalkorDB, dataset: Dataset = MOVIES) -> None:
    """Create the trial graph, replacing one of the same name."""
    graph = client.select_graph(dataset.graph)
    if dataset.graph in await client.list_graphs():
        await graph.delete()
    await graph.query(dataset.data)


@asynccontextmanager
async def trial_client(dataset: Dataset = MOVIES) -> AsyncGenerator[FalkorDB]:
    """A client on a throwaway embedded server that has the trial graph."""
    with tempfile.TemporaryDirectory() as directory:
        embedded = EmbeddedFalkorDB(str(Path(directory) / "trial.db"))
        client = FalkorDB(unix_socket_path=embedded.client.socket_file)
        try:
            await load_graph(client, dataset)
            yield client
        finally:
            await client.aclose()
            embedded.close()


def replies_for(replies_dir: Path, question_id: str) -> list[str]:
    replies: list[str] = []
    while (path := replies_dir / f"{question_id}_{len(replies) + 1}.txt").exists():
        replies.append(path.read_text(encoding="utf-8"))
    if len(list(replies_dir.glob(f"{question_id}_*.txt"))) > len(replies):
        # Replies are read in order, so a gap would hide the ones after it
        raise ValueError(
            f"{question_id}: reply {len(replies) + 1} is missing but later replies exist"
        )
    return replies


class _KeptFilter:
    """Wraps a filter and remembers what it kept, for the report."""

    def __init__(self, inner: Callable[[GraphSchema, str], Any]) -> None:
        self._inner = inner
        self.kept: GraphSchema | None = None

    async def __call__(self, schema: GraphSchema, question: str) -> FilterResult:
        kept = self._inner(schema, question)
        if inspect.isawaitable(kept):
            kept = await kept
        result = kept if isinstance(kept, FilterResult) else FilterResult(kept)
        self.kept = result.schema
        return result


async def replay(
    client: FalkorDB,
    replies_dir: Path,
    questions: Sequence[Question | Seeded],
    dataset: Dataset = MOVIES,
    condition: Condition = Condition(),  # noqa: B008 - an immutable tuple
) -> list[Pending | Finished]:
    """Run every question with the replies in ``replies_dir``, under ``condition``."""
    graph = client.select_graph(dataset.graph)
    snapshot = await Snapshot.capture_async(client, dataset.graph, types=True)
    schema = schema_from_snapshot(snapshot)
    config = Config(
        attempts_w_filtered=CONFIG.attempts_w_filtered,
        total_attempts=CONFIG.total_attempts,
        execute=CONFIG.execute,
        use_skills=condition.skills,
        example_values=condition.example_values,
    )
    items: list[Pending | Finished] = []
    for question in questions:
        replies = replies_for(replies_dir, question.id)
        llm = ReplayLLM(replies)
        kept = _KeptFilter(_filter_for(condition, llm))
        agent = Text2Cypher(
            client,
            llm,
            config=config,
            schema_filter=None if condition.filter == "none" else kept,
        )
        try:
            result = await agent.run(question.text, dataset.graph)
        except NeedReply as need:
            items.append(
                Pending(
                    question.id,
                    question.category,
                    len(replies) + 1,
                    need.system,
                    need.user,
                )
            )
            continue
        verdict = await judge(
            graph, question, result, compare_rows=dataset.compare_rows
        )
        note = filter_note(schema, question, kept.kept)
        items.append(Finished(question, result, verdict, note, len(replies)))
    return items


def _filter_for(
    condition: Condition, llm: ReplayLLM
) -> Callable[[GraphSchema, str], Any]:
    if condition.filter == "llm":
        return LLMSchemaFilter(llm)
    return filter_schema
