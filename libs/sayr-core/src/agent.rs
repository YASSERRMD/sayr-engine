use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{AgnoError, Result};
use crate::llm::{LanguageModel, ModelCompletion};
use crate::memory::ConversationMemory;
use crate::message::{Message, Role, ToolCall};
use crate::tool::ToolRegistry;

/// Structured instructions the language model should emit.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum AgentDirective {
    Respond { content: String },
    CallTool { name: String, arguments: Value },
}

/// An AGNO-style agent that alternates between the LLM and registered tools.
pub struct Agent<M: LanguageModel> {
    system_prompt: String,
    model: Arc<M>,
    tools: ToolRegistry,
    memory: ConversationMemory,
    max_steps: usize,
    streaming: bool,
}

impl<M: LanguageModel> Agent<M> {
    pub fn new(model: Arc<M>) -> Self {
        Self {
            system_prompt: "You are a helpful agent.".to_string(),
            model,
            tools: ToolRegistry::new(),
            memory: ConversationMemory::default(),
            max_steps: 6,
            streaming: false,
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_memory(mut self, memory: ConversationMemory) -> Self {
        self.memory = memory;
        self
    }

    pub fn transcript(&self) -> &[Message] {
        self.memory.messages()
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    pub fn memory(&self) -> &ConversationMemory {
        &self.memory
    }

    /// Run a single exchange with the agent. Returns the final assistant reply.
    pub async fn respond(&mut self, user_input: impl Into<String>) -> Result<String> {
        self.memory.push(Message::user(user_input));

        for _ in 0..self.max_steps {
            let mut request = vec![Message::system(self.build_system_message())];
            request.extend(self.memory.iter().cloned());
            let completion = self
                .model
                .complete_chat(&request, &self.tools.describe(), self.streaming)
                .await?;

            if !completion.tool_calls.is_empty() {
                for mut call in completion.tool_calls {
                    if call.id.is_none() {
                        call.id = Some(format!("call-{}", self.memory.len()));
                    }
                    let call_id = call.id.clone();
                    self.memory.push(Message {
                        role: Role::Assistant,
                        content: format!("Calling tool `{}`", call.name),
                        tool_call: Some(call.clone()),
                        tool_result: None,
                    });

                    let output = self.tools.call(&call.name, call.arguments.clone()).await?;
                    self.memory
                        .push(Message::tool_with_call(&call.name, output, call_id));
                }
                continue;
            }

            match completion {
                ModelCompletion {
                    content: Some(content),
                    tool_calls,
                } if tool_calls.is_empty() => {
                    self.memory.push(Message::assistant(&content));
                    return Ok(content);
                }
                _ => {
                    return Err(AgnoError::Protocol(
                        "Model response missing content and tool calls".into(),
                    ))
                }
            }
        }

        Err(AgnoError::Protocol(
            "Agent reached the step limit without returning a response".into(),
        ))
    }

    fn build_system_message(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str(&self.system_prompt);
        prompt.push_str(
            "\n\nWhen a tool is relevant, call it with JSON arguments. Otherwise, reply directly.",
        );
        if self.tools.names().is_empty() {
            prompt.push_str(" No tools are available.\n\n");
        } else {
            prompt.push_str("Available tools:\n");
            for tool in self.tools.describe() {
                prompt.push_str(&format!("- {}: {}\n", tool.name, tool.description));
                if let Some(params) = tool.parameters {
                    prompt.push_str(&format!("  parameters: {}\n", params));
                }
            }
            prompt.push('\n');
        }
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::tool::Tool;
    use crate::StubModel;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the `text` field back"
        }

        fn parameters(&self) -> Option<Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
            }))
        }

        async fn call(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn returns_llm_response_without_tools() {
        let model = StubModel::new(vec![r#"{"action":"respond","content":"Hello!"}"#.into()]);
        let mut agent = Agent::new(model);

        let reply = agent.respond("hi").await.unwrap();

        assert_eq!(reply, "Hello!");
        assert_eq!(agent.memory().len(), 2);
    }

    #[tokio::test]
    async fn executes_tool_then_replies() {
        let model = StubModel::new(vec![
            r#"{"action":"call_tool","name":"echo","arguments":{"text":"ping"}}"#.into(),
            r#"{"action":"respond","content":"Echoed your request."}"#.into(),
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let mut agent = Agent::new(model).with_tools(tools);

        let reply = agent.respond("say ping").await.unwrap();

        assert_eq!(reply, "Echoed your request.");
        assert_eq!(agent.memory().len(), 4);
    }

    #[tokio::test]
    async fn prompt_includes_tool_descriptions() {
        #[derive(Default)]
        struct RecordingModel {
            prompts: std::sync::Mutex<Vec<String>>,
        }

        #[async_trait]
        impl LanguageModel for RecordingModel {
            async fn complete_chat(
                &self,
                messages: &[Message],
                _tools: &[crate::tool::ToolDescription],
                _stream: bool,
            ) -> Result<ModelCompletion> {
                let system = messages
                    .first()
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                self.prompts.lock().unwrap().push(system);
                Ok(ModelCompletion {
                    content: Some("ok".into()),
                    tool_calls: Vec::new(),
                })
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let model = std::sync::Arc::new(RecordingModel::default());
        let mut agent = Agent::new(model.clone()).with_tools(tools);

        let _ = agent.respond("ping").await.unwrap();

        let prompts = model.prompts.lock().unwrap();
        let prompt = prompts.first().expect("prompt captured");

        assert!(prompt.contains("Echoes the `text` field back"));
        assert!(prompt.contains("Available tools"));
    }
}
