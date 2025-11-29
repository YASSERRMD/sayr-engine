//! Rust-flavored building blocks for running AGNO-style agents.
//!
//! The crate provides a minimal runtime with:
//! - A language model abstraction (`LanguageModel`).
//! - A simple tool interface (`Tool` and `ToolRegistry`).
//! - An `Agent` that loops between the model and tools using structured JSON directives.

mod agent;
mod config;
mod deployment;
mod error;
mod governance;
mod hooks;
mod knowledge;
mod llm;
mod metrics;
mod memory;
mod message;
mod server;
mod telemetry;
mod storage;
mod team;
mod tool;
mod toolkit;
mod workflow;

pub use agent::{Agent, AgentDirective};
pub use config::{AppConfig, DeploymentConfig, ModelConfig, SecurityConfig, ServerConfig, TelemetryConfig};
pub use deployment::DeploymentPlan;
pub use error::{AgnoError, Result};
pub use governance::{AccessController, Action, Principal, PrivacyRule, Role as GovernanceRole};
pub use hooks::{AgentHook, ConfirmationHandler};
pub use knowledge::{
    Document, Embedder, InMemoryVectorStore, KnowledgeBase, Retriever, ScoredDocument, VectorStore,
    WhitespaceEmbedder,
};
pub use llm::{LanguageModel, StubModel};
pub use metrics::{EvaluationReport, MetricsTracker};
pub use memory::{ConversationMemory, PersistentConversationMemory};
pub use message::{Attachment, AttachmentKind, Message, Role, ToolCall, ToolResult};
pub use server::AgentRuntime;
pub use storage::{ConversationStore, FileConversationStore, SqlConversationStore};
pub use telemetry::{FallbackChain, RetryPolicy, TelemetryCollector};
pub use team::{Team, TeamEvent};
pub use tool::{Tool, ToolRegistry};
pub use toolkit::basic_toolkit;
pub use workflow::{
    AgentTask, FunctionTask, Workflow, WorkflowContext, WorkflowNode, WorkflowTask,
};
