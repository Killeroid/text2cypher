"""The errors the library raises itself."""


class Text2CypherError(Exception):
    """The base of the errors text2cypher raises itself, so one ``except`` catches them all."""


class GraphNotFoundError(Text2CypherError, ValueError):
    """The graph does not exist, or has no labelled nodes or relationships to query."""
