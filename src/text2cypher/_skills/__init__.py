"""Short guides to the parts of FalkorDB's Cypher that a model gets wrong without them.

Each skill is a folder with a ``skill.md``: a front matter with a ``name`` and a one-line
``description``, then the guide. The folder's name is the skill's id. Adding a skill is
adding a folder; the tests run every example in every skill on a FalkorDB.
"""

import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Skill:
    """One guide.

    Attributes:
        id: The folder's name, which a model asks for it by.
        name: What it is called.
        description: One line that says what it covers and when to read it.
        body: The guide, in Markdown, with examples in ``cypher`` blocks.
    """

    id: str
    name: str
    description: str
    body: str

    @property
    def text(self) -> str:
        """What a model is given when it reads the skill."""
        return f"# {self.name}\n\n{self.body.strip()}"


@cache
def skills() -> dict[str, Skill]:
    """Every skill that ships with the package, by id, in id order."""
    return load(resources.files(__package__))


def load(directory: Traversable) -> dict[str, Skill]:
    """The skills in the folders of ``directory`` that have a ``skill.md``, by id, sorted.

    Raises:
        ValueError: If a ``skill.md`` has no front matter, or no ``name`` or ``description``,
            or a description of more than one line.
    """
    found = {}
    for folder in sorted(directory.iterdir(), key=lambda entry: entry.name):
        guide = folder / "skill.md"
        if folder.is_dir() and guide.is_file():
            found[folder.name] = _parse(folder.name, guide.read_text(encoding="utf-8"))
    return found


def _parse(skill_id: str, text: str) -> Skill:
    parts = _FRONT_MATTER.match(text)
    if parts is None:
        raise ValueError(f"The skill {skill_id!r} has no front matter")
    fields = dict(line.split(":", 1) for line in parts[1].splitlines() if ":" in line)
    name = fields.get("name", "").strip()
    description = fields.get("description", "").strip()
    if not name or not description:
        raise ValueError(f"The skill {skill_id!r} needs a name and a description")
    return Skill(skill_id, name, description, parts[2].strip())
