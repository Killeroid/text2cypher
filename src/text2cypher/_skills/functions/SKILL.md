---
name: Functions
description: Use when a question needs string, list, math, date, place, vector or statistics functions, or before you write a function you are unsure FalkorDB has. Lists the ones it has, and what it lacks or spells differently (!=, =~, datetime(), COUNT {}).
---

## Usage

Read this before you use a function you are not sure FalkorDB has, or when a question needs strings, lists, dates, places, vectors or statistics. Use the names in the examples. FalkorDB lacks some functions and syntax that other databases have; the Notes list them, so do not write them.

## Examples

Strings:

```cypher
RETURN toLower('ABC') AS lower, toUpper('abc') AS upper, substring('hello', 1, 3) AS part, split('a,b,c', ',') AS parts, replace('hello', 'l', 'L') AS replaced, trim('  x ') AS trimmed
```

```cypher
RETURN string.join(['a', 'b', 'c'], '-') AS joined, string.matchRegEx('order 123 and 456', '[0-9]+') AS matches, string.replaceRegEx('a1b2', '[0-9]', '#') AS replaced
```

Lists:

```cypher
RETURN list.dedup([1, 1, 2]) AS unique, list.sort([3, 1, 2], false) AS descending, list.insert([1, 2, 3], 1, 99) AS inserted, list.remove([1, 2, 3, 4], 1, 2) AS removed, range(1, 5) AS numbers, head([1, 2, 3]) AS first, tail([1, 2, 3]) AS rest
```

```cypher
RETURN [x IN range(1, 5) WHERE x % 2 = 0 | x * x] AS squares, reduce(total = 0, x IN [1, 2, 3] | total + x) AS total, any(x IN [1, 2] WHERE x > 1) AS some
```

Numbers, types and conversions:

```cypher
RETURN abs(-3) AS absolute, ceil(1.2) AS up, floor(1.8) AS down, round(1.5) AS rounded, sqrt(9) AS root, pow(2, 3) AS power, 7 % 3 AS remainder, toInteger('42') AS whole, toFloat('1.5') AS decimal, toJSON({a: 1, b: [1, 2]}) AS json
```

Dates, places and vectors:

```cypher
RETURN date('2024-03-05') + duration({days: 30}) AS later, date('2024-03-05').year AS year, localdatetime('2024-03-05T10:30:00') AS at, timestamp() > 0 AS now
```

```cypher
RETURN distance(point({latitude: 48.85, longitude: 2.35}), point({latitude: 52.52, longitude: 13.40})) AS meters, vec.cosineDistance(vecf32([1.0, 0.0]), vecf32([0.0, 1.0])) AS cosine
```

Aggregates and graph functions:

```cypher
MATCH (c:City)
RETURN count(c) AS n, avg(c.pop) AS average, percentileCont(c.pop, 0.5) AS median, stDev(c.pop) AS spread, collect(DISTINCT c.name) AS names
```

```cypher
MATCH (c:City {name: 'A'})
RETURN size((c)-[:ROAD]->()) AS roads, outdegree(c) AS out, indegree(c) AS inn, typeof(c.pop) AS type, coalesce(c.missing, 0) AS fallback
```

## Notes

- Not-equal is `<>`; `!=` is a syntax error.
- The regular-expression operator `=~` exists only in FalkorDB 6.0 and later; an older server rejects it. If it is rejected, use `CONTAINS`, `STARTS WITH`, `ENDS WITH` or `string.matchRegEx`.
- `COUNT { ... }` and `EXISTS { ... }` subqueries are not supported. Count relationships with `size((c)-[:ROAD]->())`, `outdegree(c)` or `indegree(c)`.
- There is no `datetime()`; use `localdatetime()`, `date()` or `timestamp()`.
- Lower case is `toLower`, not `toLowerCase`. There is no `apoc` library.
- `list.insert(list, index, value)` adds a fourth argument, `false`, to skip a value that is already in the list. `list.remove(list, index, count)` removes `count` elements from `index`. `list.sort(list, false)` sorts descending.
