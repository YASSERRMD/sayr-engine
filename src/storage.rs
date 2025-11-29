use async_trait::async_trait;
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
    pub connection_url: String,
}

impl SqlConversationStore {
    pub fn new(connection_url: impl Into<String>) -> Self {
        Self {
            connection_url: connection_url.into(),
        }
    }
}

#[async_trait]
impl ConversationStore for SqlConversationStore {
    async fn load(&self) -> Result<Vec<Message>> {
        Err(AgnoError::Storage(format!(
            "SQL backend `{}` not yet implemented",
            self.connection_url
        )))
    }

    async fn append(&self, _message: &Message) -> Result<()> {
        Err(AgnoError::Storage("SQL backend not yet implemented".into()))
    }

    async fn clear(&self) -> Result<()> {
        Err(AgnoError::Storage("SQL backend not yet implemented".into()))
    }
}
