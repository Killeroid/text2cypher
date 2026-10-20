# What is and is not checked

## What makes a query valid

A query is `valid` when its syntax is fine and none of the three validators of [cypher-guardrail](https://github.com/Killeroid/cypher-guardrail) reported an issue. Each {py:class}`~text2cypher.ValidationResult` has:

* `issues`: every issue, in the order syntax, schema, properties;
* `syntax_ok`: whether FalkorDB accepts the query;
* `valid`: `syntax_ok and not issues`;
* `schema_score` and `properties_score`: the validators' scores, kept for information. They are never used to decide validity, and are `None` when they carry no information (the query reads no property, or the syntax is invalid and that validator found nothing).

A query that can write to the graph is never valid.

## The issues

| Source | Code (after the prefix) | Meaning |
|---|---|---|
| syntax | `SyntaxError`, `ExecutionFailed`, `ArithmeticError` | FalkorDB rejects or cannot plan the query. |
| syntax | `WriteQuery` | The query can modify the graph. |
| syntax | `ParameterNotProvided` | The query uses a `$parameter` that has no value. |
| syntax | `UnboundVariable`, `UnknownFunction`, `InvalidLimit` | A variable is used but not defined, a function does not exist, or the value after `LIMIT` or `SKIP` is not a positive integer. |
| syntax | `ConflictingVariables` | One variable carries conflicting labels. |
| schema | `UnknownLabel`, `UnknownRelationshipType` | The label or relationship type is not in the graph. |
| schema | `UnknownPath` | No such path exists in the graph. |
| schema | `WrongDirectionPath` | The path exists the other way round; `suggested_path` has it. |
| schema | `VariableLengthPath`, `WrongDirectionVariableLengthPath` | The path exists with another length or direction. |
| schema | `UnknownFunction` | A call by a dotted name (`library.function(...)`) is neither a built-in function nor a user-defined function of the instance. |
| schema | `UnknownIndex` | A full-text or vector index procedure names an index that does not exist. |
| properties | `UnknownPropertyKey` | No node or relationship has this property. |
| properties | `InvalidPropertyAccess` | The label or relationship type (`kind`, `name`) does not have the property. |

Two groups of issues are reported only when you ask, with `Config(check_types=True)` and `Config(check_performance=True)`. They are heuristics, so each can report a query that is fine, and like every issue they are corrected:

| Option | Code (after the prefix) | Meaning |
|---|---|---|
| `check_types` | `PropertyTypeMismatch` | A property is compared with a value of a type none of its values has: `WHERE p.born = 'old'`. |
| `check_types` | `AggregateTypeMismatch` | `sum`, `avg`, `stDev` or the like is called on a property that holds no numbers, which FalkorDB fails on when it runs the query. |
| `check_performance` | `CartesianProduct` | Two patterns are not connected, so every row of one is paired with every row of the other. A shortest path between two nodes is one on purpose, so this option is off by default. |
| `check_performance` | `UnlabeledNodeScan` | A node with no label is matched, so every node is read. |
| `check_performance` | `UnboundedPath` | A variable-length path has no upper bound. |

Codes start with `cypher_guardrail.Warning.` or `cypher_guardrail.Error.`, except FalkorDB's own, which keep their names (`FalkorDB.ClientError.Statement.SyntaxError`). Compare against the constants of `cypher_guardrail.issues` and read the structured fields in `issue.raw`. Every issue and its fields are described in the [cypher-guardrail documentation](https://cypher-guardrail.readthedocs.io/en/latest/metadata.html).

When the LLM names an unknown label or relationship type, cypher-guardrail also reports its properties as missing. text2cypher keeps only the first, so the LLM is told about one root cause once, and it matches a repeated property issue to the unknown name by kind, so a node label and a relationship type with the same name are not mixed up.

## What is not checked

* **Meaning.** That the query answers the question that was asked. A query can be valid and still answer another one, for example by reading a property with a similar name (`rating` of a movie instead of the `stars` of a review). Look at the query, or at the rows, before you rely on them.
* **Values.** That a title or a name is spelled as it is in the data, or written in the same letter case: such a query is valid and returns no rows.
* **Performance.** Whether a query uses an index or scans the graph. `check_performance` reports the three patterns above, nothing more.
* **Result size.** Executing a valid query returns all its rows, within the time limit; a result-size limit belongs on the FalkorDB server.

These need a person, or a second model, or a test set of your own. The `evals/` folder of the repository has a small harness for the last one: it replays questions through the pipeline with replies written by a model or a person and compares the rows with those of a reference query.
