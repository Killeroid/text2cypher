"""Replay questions through ``Text2Cypher`` with replies written by someone else.

The LLM is whoever writes ``DIR/<question id>_<n>.txt``: a model, a person or a
script. ``round`` runs every question through the real ``Text2Cypher`` (the filter,
the validator, the corrector, the execution) with the replies found so far. A
question that needs another reply gets its prompt in ``DIR/pending/``; write the
reply next to the others and run ``round`` again.

From the repository root::

    uv run python -m evals.trial gold DIR             # the gold query as every first reply
    uv run python -m evals.trial seed DIR             # a flawed first query for each seeded question
    uv run python -m evals.trial round DIR [--seeded]  # replay what is in DIR and report

Add ``--dataset company`` to use the large company graph (judged by validation only).

To try a change of the pipeline add ``--filter llm`` (the schema is chosen by a model that is asked first, so
its prompt is the first reply of a question), ``--filter none``, ``--skills`` (the model may reply
``CALL read_skill {"id": "..."}`` to read a skill, as a model that cannot make a tool call stands in for one) or
``--example-values N``.

It needs the dev dependencies (the embedded FalkorDB server).
"""

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

from evals.dataset import Dataset, Question, Seeded
from evals.datasets import DATASETS
from evals.datasets.movies import MOVIES
from evals.metrics import format_metrics, summarize
from evals.replay import Condition, Finished, Pending, replay, trial_client
from text2cypher import Result


def _fenced(query: str | None) -> str:
    return f"```cypher\n{query or ''}\n```"


def _fresh_directory(path: Path) -> Path:
    """``path``, created if needed; replies left from an earlier run would be replayed."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} is not empty: use a new directory for new replies"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_replies(replies_dir: Path, replies: Mapping[str, str | None]) -> None:
    """Each reply, named ``<question id>_<n>``, as a fenced block (empty for ``None``)."""
    replies_dir = _fresh_directory(replies_dir)
    for name, query in replies.items():
        (replies_dir / f"{name}.txt").write_text(_fenced(query), encoding="utf-8")


def write_gold(replies_dir: Path, dataset: Dataset = MOVIES) -> None:
    """The gold query as the first reply to every question (an empty block to decline).

    A declined question may be asked again with the full schema, so it also gets a
    second empty reply; replies that are not needed are not used.
    """
    replies: dict[str, str | None] = {}
    for question in dataset.questions:
        replies[f"{question.id}_1"] = question.gold
        if question.gold is None:
            replies[f"{question.id}_2"] = None
    _write_replies(replies_dir, replies)


def write_seed(replies_dir: Path, dataset: Dataset = MOVIES) -> None:
    """The flawed first query of every seeded question as its first reply."""
    _write_replies(replies_dir, {f"{s.id}_1": s.flawed for s in dataset.seeded})


def prompt_text(pending: Pending) -> str:
    return (
        "You are standing in for the language model in a text-to-query pipeline.\n"
        "Below are the SYSTEM prompt and the USER message the model receives. "
        "Write the reply the model would give, exactly as it would give it, "
        f"and nothing else.\n\n=== SYSTEM ===\n{pending.system}\n=== USER ===\n{pending.user}\n"
    )


def write_pending(replies_dir: Path, items: Sequence[Pending | Finished]) -> None:
    """Put the prompt of each question that waits for a reply in ``replies_dir/pending``."""
    directory = replies_dir / "pending"
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.txt"):
        old.unlink()
    for item in items:
        if isinstance(item, Pending):
            (directory / f"{item.id}_{item.number}.txt").write_text(
                prompt_text(item), encoding="utf-8"
            )


def scopes_of(result: Result) -> str:
    """One letter per attempt: f for the filtered schema, F for the full one."""
    return "".join(
        "f" if a.schema_scope == "filtered" else "F" for a in result.attempts
    )


def format_report(items: Sequence[Pending | Finished]) -> str:
    lines: list[str] = []
    for item in items:
        if isinstance(item, Pending):
            lines.append(f"{item.id} [{item.category}] waiting for reply {item.number}")
            continue
        result, question = item.result, item.question
        scopes = scopes_of(result)
        read = [skill for attempt in result.attempts for skill in attempt.skills]
        extra = f", read {','.join(dict.fromkeys(read))}" if read else ""
        lines.append(
            f"{question.id} [{question.category}] {result.status}, "
            f"{len(result.attempts)} attempt(s) [{scopes}], {item.replies} reply(ies), "
            f"filter {item.filter_note}{extra}: "
            f"{item.verdict}\n     query: {result.query!r}"
        )
        for attempt in result.attempts[:-1]:
            for issue in attempt.validation.issues if attempt.validation else ():
                lines.append(f"     earlier issue: {issue.description[:110]}")
    finished = [item for item in items if isinstance(item, Finished)]
    waiting = len(items) - len(finished)
    if waiting:
        lines.append(f"\n{waiting} question(s) waiting for a reply")
    else:
        lines.append(
            "\nall questions finished\n\n" + format_metrics(summarize(finished))
        )
    return "\n".join(lines)


async def _round(
    replies_dir: Path, seeded: bool, dataset: Dataset, condition: Condition
) -> None:
    questions: Sequence[Question | Seeded] = (
        dataset.seeded if seeded else dataset.questions
    )
    async with trial_client(dataset) as client:
        items = await replay(client, replies_dir, questions, dataset, condition)
    write_pending(replies_dir, items)
    print(format_report(items))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["gold", "seed", "round"])
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--seeded", action="store_true", help="replay the seeded questions"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="movies")
    parser.add_argument(
        "--only", help="run only these question ids, separated by commas"
    )
    parser.add_argument(
        "--filter",
        choices=["lexical", "llm", "none"],
        default="lexical",
        help="the schema filter: the lexical one, an LLM that is asked first, or none",
    )
    parser.add_argument(
        "--skills", action="store_true", help="offer the model the skills to read"
    )
    parser.add_argument(
        "--example-values", type=int, default=3, help="example values per property"
    )
    args = parser.parse_args(argv)
    condition = Condition(args.filter, args.skills, args.example_values)
    dataset = DATASETS[args.dataset]
    if args.only:
        wanted = set(args.only.split(","))
        dataset = dataset._replace(
            questions=tuple(q for q in dataset.questions if q.id in wanted),
            seeded=tuple(q for q in dataset.seeded if q.id in wanted),
        )
    if args.command == "gold":
        write_gold(args.directory, dataset)
    elif args.command == "seed":
        write_seed(args.directory, dataset)
    else:
        asyncio.run(_round(args.directory, args.seeded, dataset, condition))


if __name__ == "__main__":
    main()
