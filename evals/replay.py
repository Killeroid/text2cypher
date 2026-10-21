"""Run questions through Text2Cypher with replies that someone else wrote."""

import tempfile
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from cypher_guardrail import Snapshot
from falkordb.asyncio import FalkorDB
from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB

from evals.dataset import Dataset, Question, Seeded
from evals.datasets.movies import MOVIES
from evals.judge import filter_note, judge
from text2cypher import Completion, Config, LLMError, Result, Text2Cypher
from text2cypher.schema import schema_from_snapshot

# One correction with the filtered schema, so the escalation can happen
CONFIG = Config(attempts_w_filtered=1, total_attempts=3, execute=True)


class NeedReply(Exception):
    """The LLM was asked something no reply has been written for yet."""

    def __init__(self, system: str, user: str) -> None:
        self.system, self.user = system, user


class ReplayLLM:
    """Answers with the replies that were written, in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._calls = 0

    async def complete(self, system: str, user: str) -> Completion:
        if self._calls == len(self._replies):
            raise NeedReply(system, user)
        self._calls += 1
        return Completion(self._replies[self._calls - 1], None)


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


async def replay(
    client: FalkorDB,
    replies_dir: Path,
    questions: Sequence[Question | Seeded],
    dataset: Dataset = MOVIES,
) -> list[Pending | Finished]:
    """Run every question with the replies in ``replies_dir``."""
    graph = client.select_graph(dataset.graph)
    snapshot = await Snapshot.capture_async(client, dataset.graph, types=True)
    schema = schema_from_snapshot(snapshot)
    items: list[Pending | Finished] = []
    for question in questions:
        replies = replies_for(replies_dir, question.id)
        agent = Text2Cypher(client, ReplayLLM(replies), config=CONFIG)
        try:
            result = await agent.run(question.text, dataset.graph)
        except LLMError as error:
            # The library wraps what the client raises; NeedReply is ours, and is the signal
            if not isinstance(need := error.__cause__, NeedReply):
                raise
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
        items.append(Finished(question, result, verdict, filter_note(schema, question)))
    return items
