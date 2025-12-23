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
mod memory;
mod message;
mod metrics;
mod server;
mod storage;
mod team;
mod telemetry;
mod tool;
mod toolkit;
mod workflow;

pub use agent::{Agent, AgentDirective};
pub use config::{
    AppConfig, DeploymentConfig, ModelConfig, ProviderConfig, SecurityConfig, ServerConfig,
    TelemetryConfig,
};
pub use deployment::DeploymentPlan;
pub use error::{AgnoError, Result};
pub use governance::{AccessController, Action, Principal, PrivacyRule, Role as GovernanceRole};
pub use hooks::{AgentHook, ConfirmationHandler};
pub use knowledge::{
    Document, DocumentChunker, Embedder, InMemoryVectorStore, KnowledgeBase, OpenAiEmbedder,
    OpenAiEmbeddingClient, PgVectorClient, PgVectorStore, QdrantClient, QdrantStore,
    RetrievalConfig, RetrievalEvaluation, RetrievalOverrides, Retriever, ScoredDocument,
    SearchParams, SimilarityMetric, SlidingWindowChunker, TransformerClient, TransformerEmbedder,
    VectorStore, WhitespaceEmbedder,
};
pub use llm::{CohereClient, LanguageModel, ModelCompletion, StubModel};
pub use memory::{ConversationMemory, PersistentConversationMemory};
pub use message::{Attachment, AttachmentKind, Message, Role, ToolCall, ToolResult};
pub use metrics::{EvaluationReport, MetricsTracker};
pub use server::AgentRuntime;
pub use storage::{ConversationStore, FileConversationStore, SqlConversationStore};
pub use team::{Team, TeamEvent};
pub use telemetry::{
    current_span_attributes, flush_tracer, init_tracing, span_with_labels, FallbackChain,
    RetryPolicy, TelemetryCollector, TelemetryLabels, TelemetrySink,
};
pub use tool::{Tool, ToolDescription, ToolRegistry};
pub use toolkit::basic_toolkit;
pub use workflow::{
    AgentTask, FunctionTask, Workflow, WorkflowContext, WorkflowNode, WorkflowTask,
};
