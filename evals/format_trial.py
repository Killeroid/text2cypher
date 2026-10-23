"""Measure the format of the prompts: plain, Markdown and XML (ADR-055), with a real model.

From the repository root, with the settings in ``.env`` (see ``evals/live.py``)::

    uv run python -m evals.format_trial smoke                # one call to each model
    uv run python -m evals.format_trial prompts              # the prompts, to read, no model
    uv run python -m evals.format_trial run --dry-run        # the whole matrix, with a stand-in model
    uv run python -m evals.format_trial run --limit 3        # the real thing on 3 questions a cell
    uv run python -m evals.format_trial run                  # the real thing
    uv run python -m evals.format_trial report               # the tables, from what is saved

Two experiments, one call to the model per question:

- ``generator``: the whole pipeline with no corrections (the format acts on the first query),
  the full schema, the data dictionary off or on, judged like any trial: the rows against the gold
  query's on ``movies``, validation alone on ``company_synonyms``.
- ``filter``: ``LLMSchemaFilter`` alone on the company schema, the data dictionary off, with the
  descriptions of labels and relationship types (B) or with those of properties too (C), judged by
  whether it kept every name the gold query needs.

Each cell (an experiment, a dataset, a model, a layout, a dictionary setting) is saved when it is
done, in ``evals/work/formats/``, and a cell that is saved is not run again unless ``--force``.
A call that fails is a row with an ``error``, never a success.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from cypher_guardrail import Snapshot

from evals.dataset import Dataset, Question
from evals.datasets import DATASETS
from evals.datasets.company import NODES, RELATIONSHIPS
from evals.datasets.company import schema as company_schema
from evals.dictionaries import COMPANY_DICTIONARY, MOVIES_DICTIONARY
from evals.formats import (
    LAYOUTS,
    Descriptions,
    Layout,
    escape,
    filter_prompts,
    filter_system,
    filter_user,
    generator_layout,
    generator_system,
    question_message,
)
from evals.judge import filter_note, judge
from evals.replay import trial_client
from text2cypher import (
    AsyncChatClient,
    AsyncLLMClient,
    Completion,
    Config,
    DataDictionary,
    LLMError,
    LLMSchemaFilter,
    Message,
    Text2Cypher,
    Usage,
)
from text2cypher.schema import GraphSchema, schema_from_snapshot

OUT = Path(__file__).parent / "work" / "formats"
type Experiment = Literal["generator", "filter"]
type Mode = Literal["off", "on", "B", "C"]
type Client = AsyncChatClient | AsyncLLMClient
MODELS = ("cheap", "expensive")

# The dictionary of each graph
DICTIONARIES: dict[str, DataDictionary] = {
    "movies": MOVIES_DICTIONARY,
    "company": COMPANY_DICTIONARY,
}
# The datasets and dictionary settings of each experiment
GENERATOR_DATASETS = ("movies", "company_synonyms")
FILTER_DATASETS = ("company_synonyms",)
GENERATOR_MODES: tuple[Mode, ...] = ("off", "on")
FILTER_MODES: tuple[Mode, ...] = ("off", "B", "C")


@dataclass(frozen=True, slots=True)
class Cell:
    """One run of a set of questions: what is varied, and nothing else."""

    experiment: Experiment
    dataset: str
    model: str
    layout: Layout
    mode: Mode

    @property
    def name(self) -> str:
        return (
            f"{self.experiment}-{self.dataset}-{self.model}-{self.layout}-{self.mode}"
        )


def matrix(
    experiments: Sequence[Experiment],
    models: Sequence[str] = MODELS,
    layouts: Sequence[Layout] = LAYOUTS,
) -> list[Cell]:
    """Every cell of ``experiments``, grouped by dataset so a database is made once for each."""
    cells = []
    for experiment in experiments:
        datasets = GENERATOR_DATASETS if experiment == "generator" else FILTER_DATASETS
        modes = GENERATOR_MODES if experiment == "generator" else FILTER_MODES
        cells += [
            Cell(experiment, dataset, model, layout, mode)
            for dataset in datasets
            for model in models
            for layout in layouts
            for mode in modes
        ]
    return cells


# --- a stand-in for the model, for a dry run and for tests ------------------------------------


class OracleLLM:
    """Answers with the gold query (or the names it needs) of the question it is asked.

    It makes the whole matrix run without a model, which shows that the prompts, the pipeline and
    the judging fit together. It says nothing about any model.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._questions = dataset.questions
        self.calls: list[tuple[str, str]] = []

    async def chat(
        self, system: str, messages: Sequence[Message], *_: Any, **__: Any
    ) -> Completion:
        user = messages[-1].content
        self.calls.append((system, user))
        # The question is in the user message, escaped in the XML layout; of two that fit, the
        # longer one is the question (one may be a part of another)
        fitting = [
            q
            for q in self._questions
            if q.text in user or escape("xml", q.text) in user
        ]
        if not fitting:
            raise ValueError("no question of the dataset is in the message")
        question = max(fitting, key=lambda q: len(q.text))
        used = Usage(len(system + user) // 4, 20)
        if "choose which parts" in system:
            words = set(re.findall(r"[A-Za-z_]\w*", question.gold or ""))
            labels = sorted(words & set(NODES))
            types = sorted(words & {kind for _, kind, _ in RELATIONSHIPS})
            reply = json.dumps({"labels": labels, "relationship_types": types})
            return Completion(f"```json\n{reply}\n```", used)
        return Completion(f"```cypher\n{question.gold or ''}\n```", used)


# --- running a cell -------------------------------------------------------------------------


def _used(usage: Usage | None) -> dict[str, int | None]:
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None}
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
    }


async def _each(
    questions: Sequence[Question],
    work: Callable[[Question], Any],
    concurrency: int,
) -> list[dict[str, Any]]:
    """``work`` on every question, a few at a time; an exception is a row, not a crash."""
    gate = asyncio.Semaphore(concurrency)

    async def one(question: Question) -> dict[str, Any]:
        async with gate:
            row = {"id": question.id, "category": question.category}
            try:
                return {**row, **await work(question)}
            except LLMError as error:
                return {
                    **row,
                    "ok": False,
                    "error": f"LLMError: {error.__cause__!r}"[:200],
                }
            except Exception as error:  # noqa: BLE001 - one question must not end a cell
                return {
                    **row,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}"[:200],
                }

    return await asyncio.gather(*(one(question) for question in questions))


async def run_generator_cell(
    cell: Cell,
    llm: Client,
    database: Any,
    questions: Sequence[Question],
    concurrency: int,
) -> list[dict[str, Any]]:
    """The pipeline on each question, with no correction, in the cell's layout."""
    dataset = DATASETS[cell.dataset]
    graph = database.select_graph(dataset.graph)
    dictionary = DICTIONARIES[dataset.graph] if cell.mode == "on" else None
    config = Config(attempts_w_filtered=0, total_attempts=0, execute=True)
    agent = Text2Cypher(database, llm, config=config, schema_filter=None)

    async def work(question: Question) -> dict[str, Any]:
        result = await agent.run(question.text, dataset.graph)
        verdict = await judge(
            graph, question, result, compare_rows=dataset.compare_rows
        )
        return {
            "ok": verdict.startswith("OK"),
            "verdict": verdict,
            "status": result.status,
            "query": result.query,
            **_used(result.usage),
        }

    with generator_layout(cell.layout, dictionary):
        return await _each(questions, work, concurrency)


async def run_filter_cell(
    cell: Cell, llm: Client, questions: Sequence[Question], concurrency: int
) -> list[dict[str, Any]]:
    """``LLMSchemaFilter`` on each question, in the cell's layout, with the dictionary as the cell says."""
    full = company_schema()
    schema = (
        full
        if cell.mode == "off"
        else replace(full, data_dictionary=COMPANY_DICTIONARY)
    )
    schema_filter = LLMSchemaFilter(
        llm, min_entities=0, property_descriptions=cell.mode == "C"
    )
    descriptions = cast(
        "Descriptions", cell.mode
    )  # "off", "B" or "C" in this experiment

    async def work(question: Question) -> dict[str, Any]:
        outcome = await schema_filter(schema, question.text)
        # The filter keeps the whole schema when its call fails, so a call that was not made
        # (no tokens counted) is an error here and not a right answer
        if outcome.usage is None or outcome.usage == Usage(0, 0):
            return {"ok": False, "error": "the model was not asked, or the call failed"}
        note = filter_note(full, question, outcome.schema)
        kept = len(outcome.schema.node_properties) + len(
            outcome.schema.relationship_properties
        )
        return {
            "ok": "MISSING" not in note,
            "verdict": note,
            "kept_names": kept,
            **_used(outcome.usage),
        }

    with filter_prompts(cell.layout, descriptions):
        return await _each(questions, work, concurrency)


async def run_cells(
    cells: Sequence[Cell],
    clients: dict[str, Client],
    *,
    out: Path,
    limit: int | None,
    concurrency: int,
    oracle: bool = False,
) -> None:
    """Run ``cells`` and save each, one database for each dataset of the generator."""
    by_dataset: dict[tuple[Experiment, str], list[Cell]] = defaultdict(list)
    for cell in cells:
        by_dataset[cell.experiment, cell.dataset].append(cell)
    for (experiment, dataset), group in by_dataset.items():
        questions = DATASETS[dataset].questions[:limit]
        if experiment == "generator":
            async with trial_client(DATASETS[dataset]) as database:
                for cell in group:
                    llm = (
                        OracleLLM(DATASETS[dataset]) if oracle else clients[cell.model]
                    )
                    rows = await run_generator_cell(
                        cell, llm, database, questions, concurrency
                    )
                    _save(out, cell, rows)
        else:
            for cell in group:
                llm = OracleLLM(DATASETS[dataset]) if oracle else clients[cell.model]
                _save(
                    out, cell, await run_filter_cell(cell, llm, questions, concurrency)
                )


def _finished(path: Path) -> bool:
    """Whether ``path`` has the rows of a cell, none of them an error: a cell with a failed
    call is run again, so a run that lost its connection does not count as done."""
    return path.exists() and not any(
        "error" in json.loads(line) for line in path.read_text("utf-8").splitlines()
    )


def unsaved(cells: Sequence[Cell], out: Path, *, force: bool) -> list[Cell]:
    """The cells that have no finished rows yet, or all of them with ``force``."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        cell for cell in cells if force or not _finished(out / f"{cell.name}.jsonl")
    ]
    print(
        f"{len(todo)} cells to run, {len(cells) - len(todo)} already saved", flush=True
    )
    return todo


def _save(out: Path, cell: Cell, rows: list[dict[str, Any]]) -> None:
    path = out / f"{cell.name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"  {cell.name}: {_counts(rows)}", flush=True)


def _counts(rows: Sequence[dict[str, Any]]) -> str:
    errors = sum("error" in row for row in rows)
    text = f"{sum(row['ok'] for row in rows)}/{len(rows)} ok"
    return text + (f", {errors} ERRORS" if errors else "")


# --- the report -----------------------------------------------------------------------------


def load_rows(out: Path) -> dict[Cell, list[dict[str, Any]]]:
    cells = {}
    for path in sorted(out.glob("*.jsonl")):
        experiment, dataset, model, layout, mode = path.stem.split("-")
        cell = Cell(
            cast("Experiment", experiment),
            dataset,
            model,
            cast("Layout", layout),
            cast("Mode", mode),
        )
        cells[cell] = [
            json.loads(line) for line in path.read_text("utf-8").splitlines()
        ]
    return cells


def _average(rows: Sequence[dict[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    return f"{sum(values) / len(values):,.0f}" if values else "-"


def report(out: Path, names: dict[str, str] | None = None) -> str:
    """The saved cells as tables: a row for each layout, a column for each dictionary setting.

    Each entry is ``right/asked`` and the mean tokens of the prompt and of the reply. A row with
    an error is not right; the number of errors is said when there are any.
    """
    cells = load_rows(out)
    names = names or {}
    tables: dict[tuple[str, str, str], dict[tuple[str, str], list[dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for cell, rows in cells.items():
        tables[cell.experiment, cell.dataset, cell.model][cell.layout, cell.mode] = rows
    lines = []
    for (experiment, dataset, model), entries in sorted(tables.items()):
        modes = sorted({mode for _, mode in entries}, key=["off", "on", "B", "C"].index)
        label = names.get(model, model)
        lines += [
            "",
            f"{experiment} / {dataset} / {model} ({label})",
            "  " + "layout".ljust(10) + "".join(m.center(26) for m in modes),
        ]
        for layout in LAYOUTS:
            cols = []
            for mode in modes:
                rows = entries.get((layout, mode))
                if rows is None:
                    cols.append("-".center(26))
                    continue
                shown = f"{_counts(rows)}  {_average(rows, 'prompt_tokens')}/{_average(rows, 'completion_tokens')} tok"
                cols.append(shown.center(26))
            lines.append("  " + layout.ljust(10) + "".join(cols))
    return "\n".join(lines).lstrip("\n")


# --- the commands -----------------------------------------------------------------------------


def _write_prompts(out: Path) -> list[Path]:
    """The prompts as they are sent, a file for each layout, for a person to read."""
    directory = out / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    async def movies_schema() -> GraphSchema:
        async with trial_client(DATASETS["movies"]) as database:
            snapshot = await Snapshot.capture_async(database, "movies", types=True)
            return schema_from_snapshot(snapshot)

    movies = replace(asyncio.run(movies_schema()), data_dictionary=MOVIES_DICTIONARY)
    company = replace(company_schema(), data_dictionary=COMPANY_DICTIONARY)
    for layout in LAYOUTS:
        path = directory / f"generator-movies-{layout}.txt"
        question = DATASETS["movies"].questions[0].text
        path.write_text(
            f"=== system ===\n{generator_system(layout, movies)}\n=== user ===\n{question_message(layout, question)}\n",
            encoding="utf-8",
        )
        written.append(path)
        path = directory / f"filter-company-{layout}.txt"
        question = DATASETS["company_synonyms"].questions[0].text
        path.write_text(
            f"=== system ===\n{filter_system(layout, dictionary=True)}\n=== user (C) ===\n"
            f"{filter_user(layout, company, question, descriptions='C')}\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


async def _smoke() -> None:
    from evals.live import load_env, make_client, model_name

    load_env()
    for which in MODELS:
        client = make_client(which)
        completion = await client.complete("Reply with the single word OK.", "ping")
        print(
            f"{which} ({model_name(which)}): reply={completion.text.strip()[:40]!r} usage={completion.usage}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="evals.format_trial", description=__doc__.splitlines()[0]
    )
    parser.add_argument("command", choices=["smoke", "prompts", "run", "report"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--experiments", default="generator,filter")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--limit", type=int, default=None, help="questions per cell")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="a stand-in model that answers with the gold query",
    )
    parser.add_argument("--force", action="store_true", help="run saved cells again")
    args = parser.parse_args(argv)

    if args.command == "smoke":
        asyncio.run(_smoke())
    elif args.command == "prompts":
        for path in _write_prompts(args.out):
            print(path)
    elif args.command == "report":
        print(report(args.out, _model_names()))
    else:
        out = args.out / "dry-run" if args.dry_run else args.out
        cells = matrix(
            args.experiments.split(","), args.models.split(","), args.layouts.split(",")
        )
        clients: dict[str, Client] = {}
        if not args.dry_run:
            from evals.live import make_client

            clients = {model: make_client(model) for model in args.models.split(",")}
        asyncio.run(
            run_cells(
                unsaved(cells, out, force=args.force),
                clients,
                out=out,
                limit=args.limit,
                concurrency=args.concurrency,
                oracle=args.dry_run,
            )
        )
        print()
        print(report(out, _model_names()))


def _model_names() -> dict[str, str]:
    """The model behind each name, for the report, when the settings can be read."""
    try:
        from evals.live import load_env, model_name

        load_env()
        return {which: model_name(which) for which in MODELS}
    except RuntimeError:
        return {}


if __name__ == "__main__":
    main(sys.argv[1:])
