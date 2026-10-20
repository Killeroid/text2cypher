import uuid
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.asyncio.connection import AbstractConnection
from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def falkordb_client(
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[FalkorDB]:
    """An async FalkorDB client on an embedded server (falkordblite), per test session.

    The sync embedded client owns the server so it is stopped cleanly with a
    save; the regular ``falkordb.asyncio`` client attaches to its socket.
    falkordblite's own ``AsyncFalkorDB`` calls ``shutdown()`` without awaiting
    it, which warns and makes it SIGKILL the server after about 5 s
    (falkordblite#61).

    Starting and stopping the server is slow, so it is shared and each test
    works in its own graph (see ``movies_graph``).
    """
    db_file = Path(tmp_path_factory.mktemp("falkordb")) / "test.db"
    embedded = EmbeddedFalkorDB(str(db_file))
    client = FalkorDB(unix_socket_path=embedded.client.socket_file)
    try:
        yield client
    finally:
        await client.aclose()
        embedded.close()


@pytest.fixture
async def movies_graph(falkordb_client: FalkorDB) -> AsyncIterator[str]:
    """Seed a uniquely named graph, return its name, and delete it afterwards."""
    name = f"movies_{uuid.uuid4().hex[:8]}"
    graph = falkordb_client.select_graph(name)
    await graph.query(
        "CREATE (:Person {name: 'Keanu'})"
        "-[:ACTED_IN {role: 'Neo'}]->"
        "(:Movie {title: 'Matrix', year: 1999})"
    )
    yield name
    await graph.delete()


FILM_GRAPH = """
CREATE (k:Person:Actor {name: 'Keanu', born: 1964, alive: true, tags: ['a']}),
       (c:Person {name: 'Carrie', born: 1967}),
       (m:Movie {title: 'Matrix', year: 1999, rating: 8.7}),
       (g:Genre {name: 'SciFi'}),
       (k)-[:ACTED_IN {role: 'Neo', since: 1999}]->(m),
       (c)-[:ACTED_IN {role: 'Trinity'}]->(m),
       (m)-[:IN_GENRE]->(g),
       (k)-[:KNOWS]->(c)
"""


@pytest.fixture
async def new_graph(falkordb_client: FalkorDB) -> AsyncIterator[AsyncGraph]:
    """A graph with a unique name and no data, deleted afterwards."""
    graph = falkordb_client.select_graph(f"schema_{uuid.uuid4().hex[:8]}")
    yield graph
    if graph.name in await falkordb_client.list_graphs():
        await graph.delete()


@pytest.fixture
async def film_graph(new_graph: AsyncGraph) -> AsyncGraph:
    await new_graph.query(FILM_GRAPH)
    return new_graph


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test that needs the embedded database, so it can be selected or skipped."""
    for item in items:
        if "falkordb_client" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """How many of each Redis command the client sends while a test runs.

    It counts at the connection, because a client binds its ``execute_command`` when it is created.
    """
    sent: Counter[str] = Counter()
    send_command = AbstractConnection.send_command

    async def counting(self: AbstractConnection, *args: Any, **kwargs: Any) -> None:
        sent[str(args[0])] += 1
        await send_command(self, *args, **kwargs)

    monkeypatch.setattr(AbstractConnection, "send_command", counting)
    return sent
