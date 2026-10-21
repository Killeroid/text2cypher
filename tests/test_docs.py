"""The documentation's own links point somewhere, and say what they point to."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The notes that are not published are not checked
INTERNAL = {"decisions.md", "development-log.md", "refactor-plan.md"}
FILES = sorted(
    [
        *(p for p in (ROOT / "docs").glob("*.md") if p.name not in INTERNAL),
        ROOT / "README.md",
        ROOT / "MIGRATION.md",
        ROOT / "CHANGELOG.md",
    ]
)
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
FENCED = re.compile(r"```.*?```", re.DOTALL)


def prose(path: Path) -> str:
    return FENCED.sub("", path.read_text(encoding="utf-8"))


def slug(heading: str) -> str:
    """How a heading becomes an anchor, as GitHub and MyST both write it."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", text))


def anchors(path: Path) -> set[str]:
    return {
        slug(match[1])
        for match in re.finditer(r"^#{1,6}\s+(.+?)\s*$", prose(path), re.MULTILINE)
    }


def internal_links(path: Path) -> list[tuple[str, str]]:
    return [
        (label, target)
        for label, target in LINK.findall(prose(path))
        if not target.startswith(("http://", "https://", "mailto:"))
    ]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_link_to_another_file_or_heading_exists(path: Path):
    for _, target in internal_links(path):
        name, _, fragment = target.partition("#")
        destination = path.parent / name if name else path
        assert destination.exists(), f"{path.name}: {target} points at no file"
        if fragment and destination.suffix == ".md":
            assert fragment in anchors(destination), (
                f"{path.name}: {target} points at no heading of {destination.name}"
            )


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_link_says_what_it_points_to(path: Path):
    # `[](page.md)` is filled in by Sphinx, and shows as nothing on GitHub and in most editors
    empty = [target for label, target in internal_links(path) if not label.strip()]
    assert not empty, f"{path.name}: links with no text: {empty}"


def test_the_published_pages_are_all_in_the_index():
    # what the site leaves out is what docs/conf.py excludes, so the two cannot drift
    conf = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    excluded = set(re.findall(r'"([\w-]+\.md)"', conf))
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^([a-z-]+)$", index, re.MULTILINE))
    pages = {p.stem for p in (ROOT / "docs").glob("*.md") if p.name not in excluded}

    assert excluded >= INTERNAL
    assert pages - {"index"} <= listed
