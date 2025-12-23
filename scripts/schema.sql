-- Conversation transcripts persisted for SQL backends.
-- This schema is compatible with the `SqlConversationStore` implemented using sqlx.
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);
