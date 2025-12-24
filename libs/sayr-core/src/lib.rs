//! Rust-flavored building blocks for running AGNO-style agents.
//!
//! The crate provides a minimal runtime with:
//! - A language model abstraction (`LanguageModel`).
//! - A simple tool interface (`Tool` and `ToolRegistry`).
//! - An `Agent` that loops between the model and tools using structured JSON directives.

mod agent;
mod error;
mod llm;
mod memory;
mod message;
mod tool;

pub use agent::{Agent, AgentDirective};
pub use error::{AgnoError, Result};
pub use llm::{LanguageModel, ModelCompletion, StubModel};
pub use memory::ConversationMemory;
pub use message::{Message, Role, ToolCall, ToolResult};
pub use tool::{Tool, ToolDescription, ToolRegistry};
