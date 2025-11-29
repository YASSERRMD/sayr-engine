//! Rust-flavored building blocks for running AGNO-style agents.
//!
//! The crate provides a minimal runtime with:
//! - A language model abstraction (`LanguageModel`).
//! - A simple tool interface (`Tool` and `ToolRegistry`).
//! - An `Agent` that loops between the model and tools using structured JSON directives.

mod agent;
mod error;
mod hooks;
mod knowledge;
mod llm;
mod memory;
mod message;
mod storage;
mod tool;
mod toolkit;

pub use agent::{Agent, AgentDirective};
pub use error::{AgnoError, Result};
pub use hooks::{AgentHook, ConfirmationHandler};
pub use knowledge::{
    Document, Embedder, InMemoryVectorStore, KnowledgeBase, Retriever, ScoredDocument, VectorStore,
    WhitespaceEmbedder,
};
pub use llm::{LanguageModel, StubModel};
pub use memory::{ConversationMemory, PersistentConversationMemory};
pub use message::{Attachment, AttachmentKind, Message, Role, ToolCall, ToolResult};
pub use storage::{ConversationStore, FileConversationStore, SqlConversationStore};
pub use tool::{Tool, ToolRegistry};
pub use toolkit::basic_toolkit;
