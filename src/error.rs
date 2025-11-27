use thiserror::Error;

pub type Result<T> = std::result::Result<T, AgnoError>;

#[derive(Debug, Error)]
pub enum AgnoError {
    #[error("tool `{0}` not found")]
    ToolNotFound(String),

    #[error("tool `{name}` invocation failed: {source}")]
    ToolInvocation {
        name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("language model error: {0}")]
    LanguageModel(String),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

