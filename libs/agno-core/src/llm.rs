use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AgnoError, Result};
use crate::message::Message;
use crate::tool::ToolDescription;

/// Result of a chat completion request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelCompletion {
    pub content: Option<String>,
    pub tool_calls: Vec<crate::message::ToolCall>,
}

/// Minimal abstraction around a chat completion provider.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion>;
}

/// A deterministic model used for tests and demos.
pub struct StubModel {
    responses: Mutex<VecDeque<String>>,
}

impl StubModel {
    pub fn new(responses: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum StubDirective {
    Respond { content: String },
    CallTool { name: String, arguments: Value },
}

#[async_trait]
impl LanguageModel for StubModel {
    async fn complete_chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDescription],
        _stream: bool,
    ) -> Result<ModelCompletion> {
        let mut locked = self.responses.lock().expect("stub model poisoned");
        let raw = locked.pop_front().ok_or_else(|| {
            AgnoError::LanguageModel("StubModel ran out of scripted responses".into())
        })?;

        match serde_json::from_str::<StubDirective>(&raw) {
            Ok(StubDirective::Respond { content }) => Ok(ModelCompletion {
                content: Some(content),
                tool_calls: Vec::new(),
            }),
            Ok(StubDirective::CallTool { name, arguments }) => Ok(ModelCompletion {
                content: None,
                tool_calls: vec![crate::message::ToolCall {
                    id: None,
                    name,
                    arguments,
                }],
            }),
            Err(_) => Ok(ModelCompletion {
                content: Some(raw),
                tool_calls: Vec::new(),
            }),
        }
    }
}
