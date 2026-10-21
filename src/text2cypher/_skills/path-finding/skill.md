---
name: Path finding
description: Use for shortest, cheapest or fastest route questions, how two nodes are connected, or paths within N hops. Covers algo.SPpaths and algo.SSpaths with weights, cost limits and path counts, variable-length patterns and where shortestPath works.
---

## Usage

Read this for "shortest path", "cheapest" or "fastest route", "how are X and Y connected" and "within N hops". Between two specific nodes use `algo.SPpaths`. From one node to the others use `algo.SSpaths`. For what is reachable in a bounded number of hops use a variable-length pattern. Always return readable fields such as names, never the bare path.

## Examples

The shortest path between two nodes:

```cypher
MATCH (a:City {name: 'A'}), (d:City {name: 'D'})
CALL algo.SPpaths({sourceNode: a, targetNode: d, relTypes: ['ROAD'], relDirection: 'outgoing', pathCount: 1})
YIELD path
RETURN [n IN nodes(path) | n.name] AS route
```

Weighted, and bounded by a cost:

```cypher
MATCH (a:City {name: 'A'}), (d:City {name: 'D'})
CALL algo.SPpaths({sourceNode: a, targetNode: d, relTypes: ['ROAD'], weightProp: 'time', costProp: 'dist', maxCost: 100, pathCount: 2})
YIELD path, pathWeight, pathCost
RETURN [n IN nodes(path) | n.name] AS route, pathWeight, pathCost
```

The best paths from one node:

```cypher
MATCH (a:City {name: 'A'})
CALL algo.SSpaths({sourceNode: a, relTypes: ['ROAD'], weightProp: 'dist', pathCount: 5})
YIELD path, pathWeight
RETURN [n IN nodes(path) | n.name] AS route, pathWeight
ORDER BY pathWeight
```

Variable-length patterns, and `shortestPath`:

```cypher
MATCH (a:City {name: 'A'})-[:ROAD*1..3]->(x)
RETURN DISTINCT x.name
ORDER BY x.name
```

```cypher
MATCH (a:City {name: 'A'}), (d:City {name: 'D'})
WITH shortestPath((a)-[:ROAD*]->(d)) AS p
RETURN length(p) AS hops
```

## Notes

- `algo.SPpaths` needs `sourceNode` and `targetNode`. `relTypes` is a list (leave it out for every type). `relDirection` is `'outgoing'` (the default), `'incoming'` or `'both'`. `maxLen` bounds the number of hops.
- `pathCount` is how many paths: `1` (the default) is the shortest, `n` is up to `n` shortest, and `0` is all of those that tie for shortest.
- `weightProp` is a relationship property to minimise the sum of, and it adds `pathWeight`. `costProp` with `maxCost` keeps only paths whose total of another property is within the limit, and adds `pathCost`.
- `algo.SSpaths` takes the same keys without `targetNode`. It returns up to `pathCount` best paths from the source to any reachable node, best first (`0` is those that tie for the best, and the default is `1`). It does not give one path per destination.
- Bound the hops of a variable-length pattern, `*1..3`, because an unbounded one can be expensive.
- `shortestPath` works only in `WITH` or `RETURN`, and `allShortestPaths` only in `MATCH`.
