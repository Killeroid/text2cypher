"""Short guides to the parts of FalkorDB's Cypher that a model gets wrong without them.

Each skill is a folder with a ``SKILL.md``: a front matter with a ``name`` and a one-line
``description``, then the guide. The folder's name is the skill's id. Adding a skill is
adding a folder; the tests run every example in every skill on a FalkorDB.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_ID = re.compile(r"[\w.-]+")
_CYPHER_BLOCK = re.compile(r"```cypher\n(.*?)```", re.DOTALL | re.IGNORECASE)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
# A clause, not a function of the same name (``list.remove``)
_WRITE = re.compile(
    r"(?<![\w.])(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH)\b", re.IGNORECASE
)

# What every skill may add up to in a prompt, for a client that cannot be offered tools and
# so is given all of them in every request
INLINE_LIMIT = 16_000

# The file of a skill's folder
GUIDE = "SKILL.md"


@dataclass(frozen=True, slots=True)
class Skill:
    """One guide.

    Attributes:
        id: The folder's name, which a model asks for it by.
        name: What it is called.
        description: One line that says what it covers and when to read it.
        body: The guide, in Markdown, with examples in ``cypher`` blocks.

    Raises:
        ValueError: If the id is empty or has anything but letters, digits, ``_``, ``-`` and
            ``.``; if the name, the description or the body is empty; if the description is
            more than one line; or if an example in a ``cypher`` block writes to the graph
            (``CREATE``, ``MERGE``, ``DELETE``, ``SET`` and the like), because the library
            only ever returns read-only queries and a guide must not teach otherwise.
    """

    id: str
    name: str
    description: str
    body: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError(
                f"The id of a skill is letters, digits, _, - and . only, not {self.id!r}"
            )
        if (
            not self.name.strip()
            or not self.description.strip()
            or not self.body.strip()
        ):
            raise ValueError(
                f"The skill {self.id!r} needs a name, a description and a body"
            )
        if "\n" in self.description.strip():
            raise ValueError(
                f"The description of the skill {self.id!r} is more than one line"
            )
        for block in _CYPHER_BLOCK.findall(self.body):
            written = _WRITE.search(_STRING.sub("", block))
            if written:
                raise ValueError(
                    f"The skill {self.id!r} has an example that writes ({written[1].upper()}); "
                    "an example must be read-only"
                )

    @property
    def text(self) -> str:
        """What a model is given when it reads the skill."""
        return f"# {self.name}\n\n{self.body.strip()}"


@cache
def skills() -> dict[str, Skill]:
    """Every skill that ships with the package, by id, in id order."""
    return load(resources.files(__package__))


def builtin_skills() -> list[Skill]:
    """The skills that ship with the package, in id order."""
    return list(skills().values())


def load_skills(directory: str | Path) -> list[Skill]:
    """The skills in the folders of ``directory``, as ``load`` reads them, in id order.

    Raises:
        ValueError: If a ``SKILL.md`` is not a valid skill, or there is none.
        FileNotFoundError: If ``directory`` does not exist.
    """
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    found = list(load(path).values())
    if not found:
        raise ValueError(f"There is no skill (a folder with a SKILL.md) in {path}")
    return found


def catalog(chosen: Iterable[Skill]) -> dict[str, Skill]:
    """``chosen`` by id, in the order given.

    Raises:
        ValueError: If there are none, or two have the same id.
    """
    found: dict[str, Skill] = {}
    for skill in chosen:
        if skill.id in found:
            raise ValueError(f"Two skills have the id {skill.id!r}")
        found[skill.id] = skill
    if not found:
        raise ValueError("skills is empty; turn use_skills off instead")
    return found


def check_inline(guides: Mapping[str, Skill]) -> None:
    """Raise ``ValueError`` if ``guides`` are too long to put in every prompt."""
    size = sum(len(skill.text) for skill in guides.values())
    if size > INLINE_LIMIT:
        raise ValueError(
            f"The skills add up to {size} characters, more than the {INLINE_LIMIT} that fit in "
            "every prompt of a client that cannot be offered tools; use fewer or shorter ones"
        )


def load(directory: Traversable) -> dict[str, Skill]:
    """The skills in the folders of ``directory`` that have a ``SKILL.md``, by id, sorted.

    The file is found whatever the case of its name (``SKILL.md``, ``skill.md``), the same
    on every file system, because the entries are listed and compared, not opened by name.

    Raises:
        ValueError: If a ``SKILL.md`` has no front matter, or no ``name`` or ``description``,
            or a description of more than one line, or a folder has two files whose names
            differ only in case.
    """
    found = {}
    for folder in sorted(directory.iterdir(), key=lambda entry: entry.name):
        if not folder.is_dir():
            continue
        guides = [
            entry
            for entry in folder.iterdir()
            if entry.name.lower() == GUIDE.lower() and entry.is_file()
        ]
        if len(guides) > 1:
            raise ValueError(f"The skill {folder.name!r} has more than one {GUIDE}")
        if guides:
            found[folder.name] = _parse(
                folder.name, guides[0].read_text(encoding="utf-8")
            )
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
