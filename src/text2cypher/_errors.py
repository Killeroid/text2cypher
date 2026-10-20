"""The errors the library raises itself."""


class GraphNotFoundError(ValueError):
    """The graph does not exist, or has no labelled nodes or relationships to query."""
