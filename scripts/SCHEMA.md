# Conversation storage schema

The SQL conversation store uses a single `messages` table to capture serialized transcript entries. The schema is defined in [`schema.sql`](./schema.sql) and can be applied with any SQLite-compatible migration tool:

```bash
sqlite3 conversation.db < scripts/schema.sql
```

For other SQLx-supported backends, adapt the DDL to your target database.
