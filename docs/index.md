# text2cypher

Ask a question in plain language and get back a read-only Cypher query that has been checked against a [FalkorDB](https://www.falkordb.com/) graph.

text2cypher asks an LLM to write the query, validates it against the real graph with [cypher-guardrail](https://github.com/Killeroid/cypher-guardrail), and sends what is wrong back to the LLM until the query passes or the attempts run out. A query that fails the checks is never handed back as if it were fine.

```{toctree}
:maxdepth: 2
:caption: Guides

usage
how-it-works
schema
llm
checks
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference
changelog
```
