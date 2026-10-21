"""Questions about paths, graph algorithms and functions, on a small graph with real data.

They measure whether offering the skills helps. Each has the query that answers it, run on the
graph and compared by its rows. A plain question is in the set, to see that the skills do not hurt it. The graph: people who know each other (two
separate groups), and cities joined by roads with a distance and a time.
"""

from evals.dataset import Dataset, Question

DATA = """
CREATE (alice:Person {name: 'Alice'}), (bob:Person {name: 'Bob'}), (carol:Person {name: 'Carol'}),
       (dave:Person {name: 'Dave'}), (eve:Person {name: 'Eve'}), (frank:Person {name: 'Frank'}),
       (grace:Person {name: 'Grace'}),
       (alice)-[:KNOWS]->(bob), (bob)-[:KNOWS]->(carol), (alice)-[:KNOWS]->(carol),
       (carol)-[:KNOWS]->(dave), (eve)-[:KNOWS]->(dave), (frank)-[:KNOWS]->(grace),
       (a:City {name: 'A', pop: 10}), (b:City {name: 'B', pop: 20}), (c:City {name: 'C', pop: 30}),
       (d:City {name: 'D', pop: 40}), (e:City {name: 'E', pop: 50}),
       (a)-[:ROAD {dist: 1, time: 5}]->(b), (b)-[:ROAD {dist: 1, time: 5}]->(c),
       (a)-[:ROAD {dist: 5, time: 1}]->(c), (c)-[:ROAD {dist: 2, time: 2}]->(d),
       (e)-[:ROAD {dist: 1, time: 1}]->(d)
"""

QUESTIONS = (
    Question(
        "k01",
        "shortest path by a weight",
        "What is the shortest route by distance from city A to city D? Give the route as a list of city names.",
        "MATCH (a:City {name: 'A'}), (d:City {name: 'D'}) "
        "CALL algo.SPpaths({sourceNode: a, targetNode: d, relTypes: ['ROAD'], weightProp: 'dist'}) "
        "YIELD path RETURN [n IN nodes(path) | n.name] AS route",
    ),
    Question(
        "k02",
        "centrality",
        "Who is the most important person in the KNOWS network?",
        "CALL algo.pageRank('Person', 'KNOWS') YIELD node, score "
        "RETURN node.name ORDER BY score DESC LIMIT 1",
    ),
    Question(
        "k03",
        "connected groups",
        "How many separate groups of connected people are there?",
        "CALL algo.WCC({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) "
        "YIELD node, componentId RETURN count(DISTINCT componentId)",
    ),
    Question(
        "k04",
        "a statistic",
        "What is the median population of the cities?",
        "MATCH (c:City) RETURN percentileCont(c.pop, 0.5)",
    ),
    Question(
        "k05",
        "list functions",
        "List the distinct city populations from highest to lowest, as one list.",
        "MATCH (c:City) RETURN list.sort(collect(DISTINCT c.pop), false)",
    ),
    Question(
        "k06",
        "plain, a control",
        "Which cities have a population above 25?",
        "MATCH (c:City) WHERE c.pop > 25 RETURN c.name",
    ),
    Question(
        "k07",
        "shortest path by another weight",
        "What is the quickest route by travel time from city A to city D? Give the route as a list of city names.",
        "MATCH (a:City {name: 'A'}), (d:City {name: 'D'}) "
        "CALL algo.SPpaths({sourceNode: a, targetNode: d, relTypes: ['ROAD'], weightProp: 'time'}) "
        "YIELD path RETURN [n IN nodes(path) | n.name] AS route",
    ),
    Question(
        "k08",
        "traversal",
        "How many cities can be reached from city A by following roads, counting A itself (so A counts as one)?",
        "MATCH (a:City {name: 'A'}) CALL algo.BFS(a, -1, 'ROAD') YIELD nodes RETURN size(nodes) + 1",
    ),
    Question(
        "k09",
        "plain, a control",
        "What is the average population of the cities?",
        "MATCH (c:City) RETURN avg(c.pop)",
    ),
    Question(
        "k10",
        "a statistic",
        "What is the 90th percentile of the city populations, taking an actual value from the list?",
        "MATCH (c:City) RETURN percentileDisc(c.pop, 0.9)",
    ),
    Question(
        "k11",
        "plain but looks special",
        "What is the shortest chain of people from Alice to Dave through KNOWS, as a list of names?",
        "MATCH (a:Person {name: 'Alice'}), (d:Person {name: 'Dave'}) "
        "RETURN [n IN nodes(shortestPath((a)-[:KNOWS*]->(d))) | n.name]",
    ),
    Question(
        "k12",
        "list functions",
        "Give the city names as one list, sorted from Z to A.",
        "MATCH (c:City) RETURN list.sort(collect(c.name), false)",
    ),
    Question(
        "k13",
        "plain, a control",
        "What is the total population of all the cities?",
        "MATCH (c:City) RETURN sum(c.pop)",
    ),
    Question(
        "k14",
        "string and list functions",
        "Give the city names in lower case, sorted from A to Z, joined into one string with commas and no spaces.",
        "MATCH (c:City) RETURN string.join(list.sort(collect(toLower(c.name)), true), ',')",
    ),
    Question(
        "k15",
        "plain, a control",
        "How many roads leave each city? Give the city name and the number.",
        "MATCH (c:City) OPTIONAL MATCH (c)-[r:ROAD]->() RETURN c.name, count(r)",
    ),
)

SKILLS = Dataset("skills", DATA, QUESTIONS, ())
