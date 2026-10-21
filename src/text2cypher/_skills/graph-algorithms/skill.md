---
name: Graph algorithms
description: Use for questions about importance, influence, centrality, groups, communities, connected components or what is reachable within N steps. Covers PageRank, betweenness, WCC, label propagation and BFS (algo.*), with their exact arguments.
---

## Usage

Read this for questions about importance, centrality, groups or reachability: "most important", "most influential", "most central", "which groups", "connected components", "communities", "reachable within N steps". Call the procedure with `CALL ... YIELD`, then `RETURN` the properties of `node` that the question asks for, ordered by `score` when it ranks. For ordinary counts and lookups, use a plain `MATCH`.

## Examples

PageRank, the most important nodes:

```cypher
CALL algo.pageRank('Person', 'KNOWS') YIELD node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 3
```

Betweenness, the nodes that connect others:

```cypher
CALL algo.betweenness({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) YIELD node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 3
```

Connected components, and communities:

```cypher
CALL algo.WCC({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) YIELD node, componentId
RETURN componentId, count(node) AS size
ORDER BY size DESC
```

```cypher
CALL algo.labelPropagation({nodeLabels: ['Person'], relationshipTypes: ['KNOWS']}) YIELD node, communityId
RETURN communityId, collect(node.name) AS members
```

Breadth-first search, what is reachable:

```cypher
MATCH (p:Person {name: 'Alice'})
CALL algo.BFS(p, 2, 'KNOWS') YIELD nodes
RETURN [n IN nodes | n.name] AS reached
```

## Notes

- `algo.pageRank(label, relationshipType)` takes the label and the type as two strings, not a map. It yields `node` and `score`.
- `algo.betweenness`, `algo.WCC` and `algo.labelPropagation` take one optional map with `nodeLabels` and `relationshipTypes`; with no map they use the whole graph. They yield `node` with `score`, `componentId` and `communityId` respectively.
- `algo.WCC` ignores the direction of relationships.
- `algo.BFS(start, maxDepth, relationshipType)`: use `-1` for no depth limit and `null` for any type. It follows the direction of the relationships, its `nodes` leave out the start node, and it yields `nodes` and `edges` in one row.
- All of these are read-only.
