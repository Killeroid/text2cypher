# API reference

The public API is what `text2cypher`, `text2cypher.llm`, `text2cypher.schema` and `text2cypher.skills` export. Modules whose name starts with an underscore are internal and may change without notice.

## Asking a question

```{eval-rst}
.. autoclass:: text2cypher.Text2Cypher
   :members:
.. autoclass:: text2cypher.Config
.. autoexception:: text2cypher.Text2CypherError
.. autoexception:: text2cypher.GraphNotFoundError
.. autoexception:: text2cypher.LLMError
   :members:
.. autoexception:: text2cypher.DatabaseError
```

## The result

```{eval-rst}
.. autoclass:: text2cypher.Result
   :members:
.. autoclass:: text2cypher.Attempt
.. autoclass:: text2cypher.Records
   :members:
.. autoclass:: text2cypher.GraphNode
   :members:
.. autoclass:: text2cypher.GraphRelationship
   :members:
.. autoclass:: text2cypher.GraphPath
.. autofunction:: text2cypher.to_python
.. autoclass:: text2cypher.Issue
   :members:
.. autoclass:: text2cypher.ValidationResult
   :members:
.. autodata:: text2cypher._validation.issues.Source
   :no-value:
.. autodata:: text2cypher._results.results.Status
   :no-value:
.. autodata:: text2cypher._results.results.SchemaScope
   :no-value:
```

## LLM clients

```{eval-rst}
.. autoclass:: text2cypher.AsyncChatClient
   :members:
.. autoclass:: text2cypher.AsyncLLMClient
   :members:
.. autoclass:: text2cypher.Message
.. autoclass:: text2cypher.Tool
.. autoclass:: text2cypher.ToolCall
.. autoclass:: text2cypher.Completion
.. autoclass:: text2cypher.Usage
.. autoclass:: text2cypher.ModelOptions
   :members:
.. autoclass:: text2cypher.FakeLLM
   :members:
.. autoclass:: text2cypher.llm.Call
```

## Skills

```{eval-rst}
.. autoclass:: text2cypher.Skill
.. autofunction:: text2cypher.skills.builtin_skills
.. autofunction:: text2cypher.skills.load_skills
```

## The schema

```{eval-rst}
.. autoclass:: text2cypher.GraphSchema
.. autoclass:: text2cypher.schema.Path
.. autoclass:: text2cypher.schema.Index
.. autodata:: text2cypher.schema._filter.SchemaFilter
   :no-value:
.. autofunction:: text2cypher.filter_schema
.. autofunction:: text2cypher.schema.restrict_schema
.. autoclass:: text2cypher.DataDictionary
   :members: nodes, relationships, only, check
.. autoclass:: text2cypher.schema._dictionary.Entry
.. autoclass:: text2cypher.LLMSchemaFilter
   :special-members: __call__
.. autoclass:: text2cypher.FilterResult
.. autoclass:: text2cypher.AsyncSchemaFilter
   :special-members: __call__
.. autofunction:: text2cypher.schema.render_schema
.. autofunction:: text2cypher.schema.schema_from_snapshot
```
