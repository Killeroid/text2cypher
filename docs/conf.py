"""Sphinx configuration for the text2cypher documentation."""

from importlib.metadata import version as _version

project = "text2cypher"
author = "Kwaku Yeboah-Antwi"
copyright = "2026, Kwaku Yeboah-Antwi"
release = _version("text2cypher")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
# Notes for the people who work on the project, kept next to the guides but not published
exclude_patterns = [
    "_build",
    "architecture.md",
    "decisions.md",
    "development-log.md",
    "refactor-plan.md",
]

# The guides are Markdown; link to `#anchors` of headings down to ### from other pages
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Every type in a signature must resolve, so a renamed or missing one fails the build.
# These third-party types come from projects that do not publish an object inventory.
nitpicky = True
nitpick_ignore_regex = [
    (r"py:class", r"openai\..*"),
    (r"py:class", r"anthropic\..*"),
    (r"py:class", r"Async(OpenAI|Anthropic)"),
    (r"py:class", r"falkordb\..*"),
    (r"py:class", r"cypher_guardrail\..*"),
]

html_theme = "furo"
html_title = f"text2cypher {release}"
