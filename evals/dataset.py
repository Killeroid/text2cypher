"""What a trial runs on: a graph, and the questions asked of it."""

from typing import NamedTuple


class Question(NamedTuple):
    id: str
    category: str
    text: str
    # A query with the answer, or None when the right outcome is that the LLM declines
    gold: str | None


class Seeded(NamedTuple):
    """A flawed first query, written as the LLM's first reply, for it to correct."""

    id: str
    category: str
    text: str
    flawed: str
    gold: str | None


class Dataset(NamedTuple):
    """A graph with its questions: what a trial runs on."""

    graph: str
    data: str
    questions: tuple[Question, ...]
    seeded: tuple[Seeded, ...]
    # Whether a final query is judged by the rows it returns (the graph has real data)
    # or only by passing validation (the graph only has the shape of a schema)
    compare_rows: bool = True
