"""The numbers of a trial: what was answered, what the corrector healed, what went wrong."""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from evals.replay import Finished


@dataclass(frozen=True, slots=True)
class Metrics:
    """Counts over the finished questions of one trial.

    Attributes:
        answerable: Questions that have a gold query.
        correct: Answerable questions whose final query returned the gold rows.
        first_try: Answerable questions whose first query passed validation and was right.
        needed_correction: Answerable questions whose first query failed validation.
        healed: Of those, the ones that ended right.
        corrections_to_heal: The corrections the healed questions used, in all.
        declinable: Questions that should be declined (no gold query).
        declined: Of those, the ones the LLM declined.
        issues_by_code: How often each validation issue code was reported, over every attempt.
    """

    answerable: int
    correct: int
    first_try: int
    needed_correction: int
    healed: int
    corrections_to_heal: int
    declinable: int
    declined: int
    issues_by_code: dict[str, int]

    @property
    def success_rate(self) -> float | None:
        return self.correct / self.answerable if self.answerable else None

    @property
    def self_healing_rate(self) -> float | None:
        return self.healed / self.needed_correction if self.needed_correction else None


def _right(item: Finished) -> bool:
    return item.verdict.startswith("OK")


def _first_query_failed(item: Finished) -> bool:
    """Whether the first query was written and did not pass validation."""
    first = item.result.attempts[0].validation
    return first is not None and not first.valid


def _issue_codes(items: Sequence[Finished]) -> dict[str, int]:
    """How often each issue code was reported, over every attempt."""
    return dict(
        Counter(
            issue.code
            for item in items
            for attempt in item.result.attempts
            if attempt.validation
            for issue in attempt.validation.issues
        )
    )


def _first_try(item: Finished) -> bool:
    return _right(item) and len(item.result.attempts) == 1


def _healed(item: Finished) -> bool:
    return _first_query_failed(item) and _right(item)


def _count(items: Iterable[Finished], test: Callable[[Finished], bool]) -> int:
    return sum(1 for item in items if test(item))


def summarize(items: Sequence[Finished]) -> Metrics:
    answerable = [item for item in items if item.question.gold is not None]
    declinable = [item for item in items if item.question.gold is None]
    healed = [item for item in answerable if _healed(item)]
    return Metrics(
        answerable=len(answerable),
        correct=_count(answerable, _right),
        first_try=_count(answerable, _first_try),
        needed_correction=_count(answerable, _first_query_failed),
        healed=len(healed),
        corrections_to_heal=sum(len(item.result.attempts) - 1 for item in healed),
        declinable=len(declinable),
        declined=_count(declinable, _right),
        issues_by_code=_issue_codes(items),
    )


def format_metrics(metrics: Metrics) -> str:
    def rate(part: int, whole: int) -> str:
        return f"{part}/{whole}" + (f" ({part / whole:.0%})" if whole else "")

    lines = [
        f"answered correctly: {rate(metrics.correct, metrics.answerable)}",
        f"right on the first query: {rate(metrics.first_try, metrics.answerable)}",
        f"healed by correction: {rate(metrics.healed, metrics.needed_correction)}"
        + (
            f", {metrics.corrections_to_heal / metrics.healed:.1f} corrections each"
            if metrics.healed
            else ""
        ),
        f"declined as expected: {rate(metrics.declined, metrics.declinable)}",
    ]
    if metrics.issues_by_code:
        lines.append("issues reported:")
        lines.extend(
            f"  {count:3d}  {code}"
            for code, count in sorted(
                metrics.issues_by_code.items(), key=lambda pair: (-pair[1], pair[0])
            )
        )
    return "\n".join(lines)
