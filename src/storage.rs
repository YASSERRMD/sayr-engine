use async_trait::async_trait;
use sqlx::{sqlite::SqlitePoolOptions, Row, SqlitePool};
use tokio::{fs, io::AsyncWriteExt};

use crate::error::{AgnoError, Result};
use crate::message::Message;

/// Generic persistence contract for conversation state.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    async fn load(&self) -> Result<Vec<Message>>;
    async fn append(&self, message: &Message) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}

/// A simple JSONL-based store that writes messages to disk.
pub struct FileConversationStore {
    path: String,
}

impl FileConversationStore {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl ConversationStore for FileConversationStore {
    async fn load(&self) -> Result<Vec<Message>> {
        let content = match fs::read_to_string(&self.path).await {
            Ok(contents) => contents,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(AgnoError::Storage(format!(
                    "failed to read transcript `{}`: {err}",
                    self.path
                )))
            }
        };

        let mut messages = Vec::new();
        for line in content.lines() {
            let msg: Message = serde_json::from_str(line)?;
            messages.push(msg);
        }

        Ok(messages)
    }

    async fn append(&self, message: &Message) -> Result<()> {
        let mut serialized = serde_json::to_string(message)?;
        serialized.push('\n');
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(|err| {
                AgnoError::Storage(format!("failed to open `{}`: {err}", self.path.clone()))
            })?
            .write_all(serialized.as_bytes())
            .await
            .map_err(|err| AgnoError::Storage(format!("failed to persist message: {err}")))
    }

    async fn clear(&self) -> Result<()> {
        fs::remove_file(&self.path)
            .await
            .or_else(|err| {
                if err.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(err)
                }
            })
            .map_err(|err| AgnoError::Storage(format!("failed clearing `{}`: {err}", self.path)))
    }
}

/// Placeholder for SQL-based backends. The type compiles without requiring the
/// database drivers and can be swapped out once the feature lands.
pub struct SqlConversationStore {
    pool: SqlitePool,
}

impl SqlConversationStore {
    const INIT_STATEMENT: &'static str = r#"
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL
        )
    "#;

    pub async fn connect(connection_url: impl AsRef<str>) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(connection_url.as_ref())
            .await
            .map_err(|err| {
                AgnoError::Storage(format!(
                    "failed connecting to SQL backend `{}`: {err}",
                    connection_url.as_ref()
                ))
            })?;

        sqlx::query(Self::INIT_STATEMENT)
            .execute(&pool)
            .await
            .map_err(|err| AgnoError::Storage(format!("failed initializing schema: {err}")))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl ConversationStore for SqlConversationStore {
    async fn load(&self) -> Result<Vec<Message>> {
        let rows = sqlx::query("SELECT payload FROM messages ORDER BY id ASC")
            .fetch_all(&self.pool)
            .await
            .map_err(|err| AgnoError::Storage(format!("failed loading messages: {err}")))?;

        rows.into_iter()
            .map(|row| {
                let payload: String = row.try_get("payload").map_err(|err| {
                    AgnoError::Storage(format!("failed decoding message payload: {err}"))
                })?;
                serde_json::from_str(&payload)
                    .map_err(|err| AgnoError::Storage(format!("invalid message payload: {err}")))
            })
            .collect()
    }

    async fn append(&self, message: &Message) -> Result<()> {
        let payload = serde_json::to_string(message)?;
        sqlx::query("INSERT INTO messages (payload) VALUES (?)")
            .bind(payload)
            .execute(&self.pool)
            .await
            .map(|_| ())
            .map_err(|err| AgnoError::Storage(format!("failed writing message: {err}")))
    }

    async fn clear(&self) -> Result<()> {
        sqlx::query("DELETE FROM messages")
            .execute(&self.pool)
            .await
            .map(|_| ())
            .map_err(|err| AgnoError::Storage(format!("failed clearing messages: {err}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn file_store_round_trip() {
        let file = NamedTempFile::new().unwrap();
        let store = FileConversationStore::new(file.path().to_str().unwrap());

        let msg = Message::user("hello");
        store.append(&msg).await.unwrap();

        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, Role::User);

        store.clear().await.unwrap();
        let cleared = store.load().await.unwrap();
        assert!(cleared.is_empty());
    }

    #[tokio::test]
    async fn sqlite_store_round_trip() {
        let store = SqlConversationStore::connect("sqlite::memory:")
            .await
            .unwrap();

        let msg = Message::assistant("hi from db");
        store.append(&msg).await.unwrap();

        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "hi from db");

        store.clear().await.unwrap();
        let cleared = store.load().await.unwrap();
        assert!(cleared.is_empty());
    }
}
