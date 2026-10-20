"""Ask a question of a graph without a model or a server.

It starts an embedded FalkorDB (FalkorDBLite) with a small film graph and answers with a
scripted LLM, so the whole flow, including a correction, runs in a few seconds:

    uv run python examples/quickstart.py
"""

import asyncio
import tempfile
from pathlib import Path

from falkordb.asyncio import FalkorDB
from redislite.falkordb_client import FalkorDB as EmbeddedFalkorDB

from text2cypher import Config, FakeLLM, Text2Cypher

GRAPH = """
CREATE (keanu:Person {name: 'Keanu Reeves'}),
       (carrie:Person {name: 'Carrie-Anne Moss'}),
       (matrix:Movie {title: 'The Matrix', year: 1999}),
       (keanu)-[:ACTED_IN {role: 'Neo'}]->(matrix),
       (carrie)-[:ACTED_IN {role: 'Trinity'}]->(matrix)
"""

# What the model answers: the first query has the relationship the wrong way round, the
# second is its correction. A real model is asked the same way, through an LLM client.
REPLIES = [
    "```cypher\nMATCH (m:Movie {title: 'The Matrix'})-[:ACTED_IN]->(p:Person) RETURN p.name\n```",
    "```cypher\nMATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name\n```",
]


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        embedded = EmbeddedFalkorDB(str(Path(directory) / "movies.db"))
        client = FalkorDB(unix_socket_path=embedded.client.socket_file)
        try:
            await client.select_graph("movies").query(GRAPH)

            agent = Text2Cypher(client, FakeLLM(REPLIES), config=Config(execute=True))
            result = await agent.run(
                "Which actors appeared in The Matrix?", graph="movies"
            )

            print(f"status: {result.status}")
            for number, attempt in enumerate(result.attempts, start=1):
                print(f"attempt {number}: {attempt.query}")
                for issue in attempt.validation.issues if attempt.validation else ():
                    print(f"  problem: {issue.description}")
            print(f"columns: {result.records.columns if result.records else None}")
            print(f"rows: {result.records.rows if result.records else None}")
        finally:
            await client.aclose()
            embedded.close()


if __name__ == "__main__":
    asyncio.run(main())
