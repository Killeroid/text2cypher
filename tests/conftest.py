import os
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

import pytest
from docker.errors import DockerException
from falkordb.asyncio import FalkorDB
from falkordb.asyncio.graph import AsyncGraph
from redis.asyncio.connection import AbstractConnection
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

FALKORDB_IMAGE = "falkordb/falkordb:6.0.3"


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def falkordb_client() -> AsyncIterator[FalkorDB]:
    """An async FalkorDB client on a real server, per test session.

    The server is the ``FALKORDB_IMAGE`` Docker image, started on a free port and
    removed afterwards. Set ``FALKORDB_HOST`` (and ``FALKORDB_PORT``, default 6379)
    to use a server that is already running, as CI does.

    Starting and stopping the server is slow, so it is shared and each test
    works in its own graph (see ``movies_graph``).
    """
    if host := os.environ.get("FALKORDB_HOST"):
        client = FalkorDB(host=host, port=int(os.environ.get("FALKORDB_PORT", "6379")))
        try:
            yield client
        finally:
            await client.aclose()
        return
    try:
        container = (
            DockerContainer(FALKORDB_IMAGE)
            .with_exposed_ports(6379)
            .waiting_for(LogMessageWaitStrategy("Ready to accept connections"))
        )
        container.start()
    except DockerException as error:
        pytest.fail(
            "The tests need a FalkorDB server: start Docker, or set FALKORDB_HOST "
            f"(and FALKORDB_PORT) to one that is running. {error}",
            pytrace=False,
        )
    try:
        client = FalkorDB(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(6379)),
        )
        try:
            yield client
        finally:
            await client.aclose()
    finally:
        container.stop()


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
    """Mark every test that needs the FalkorDB server, so it can be selected or skipped."""
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
