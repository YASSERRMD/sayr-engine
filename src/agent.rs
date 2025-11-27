use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{AgnoError, Result};
use crate::llm::LanguageModel;
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
}

impl<M: LanguageModel> Agent<M> {
    pub fn new(model: Arc<M>) -> Self {
        Self {
            system_prompt: "You are a helpful agent.".to_string(),
            model,
            tools: ToolRegistry::new(),
            memory: ConversationMemory::default(),
            max_steps: 6,
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

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
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
            let prompt = self.build_prompt();
            let raw = self.model.complete(&prompt).await?;
            let directive: AgentDirective = serde_json::from_str(&raw).map_err(|err| {
                AgnoError::Protocol(format!(
                    "Expected JSON directive with `action`, got `{raw}`: {err}"
                ))
            })?;

            match directive {
                AgentDirective::Respond { content } => {
                    self.memory.push(Message::assistant(&content));
                    return Ok(content);
                }
                AgentDirective::CallTool { name, arguments } => {
                    self.memory.push(Message {
                        role: Role::Assistant,
                        content: format!("Calling tool `{name}`"),
                        tool_call: Some(ToolCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        }),
                        tool_result: None,
                    });

                    let output = self.tools.call(&name, arguments).await?;
                    self.memory.push(Message::tool(&name, output));
                }
            }
        }

        Err(AgnoError::Protocol(
            "Agent reached the step limit without returning a response".into(),
        ))
    }

    fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str(&format!("System: {}\n\n", self.system_prompt));
        prompt.push_str("You must answer with JSON in one of the following formats:\n");
        prompt.push_str("- {\"action\":\"respond\",\"content\":\"<final assistant message>\"}\n");
        prompt.push_str("- {\"action\":\"call_tool\",\"name\":\"<tool name>\",\"arguments\":{...}}\n\n");
        if self.tools.names().is_empty() {
            prompt.push_str("No tools are available.\n\n");
        } else {
            prompt.push_str("Available tools:\n");
            for name in self.tools.names() {
                prompt.push_str(&format!("- {name}\n"));
            }
            prompt.push('\n');
        }

        prompt.push_str("Conversation so far:\n");
        for message in self.memory.iter() {
            prompt.push_str(&format!("[{:?}] {}\n", message.role, message.content));
            if let Some(call) = &message.tool_call {
                prompt.push_str(&format!("  -> calling {} with {}\n", call.name, call.arguments));
            }
            if let Some(result) = &message.tool_result {
                prompt.push_str(&format!("  <- {} returned {}\n", result.name, result.output));
            }
        }

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::tool::Tool;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes the `text` field back"
        }

        async fn call(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn returns_llm_response_without_tools() {
        let model = StubModel::new(vec![
            r#"{"action":"respond","content":"Hello!"}"#.into(),
        ]);
        let mut agent = Agent::new(model);

        let reply = agent.respond("hi").await.unwrap();

        assert_eq!(reply, "Hello!");
        assert_eq!(agent.memory().len(), 2);
    }

    #[tokio::test]
    async fn executes_tool_then_replies() {
        let model = StubModel::new(vec![
            r#"{"action":"respond","content":"Echoed your request."}"#.into(),
            r#"{"action":"call_tool","name":"echo","arguments":{"text":"ping"}}"#.into(),
        ]);
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);

        let mut agent = Agent::new(model).with_tools(tools);

        let reply = agent.respond("say ping").await.unwrap();

        assert_eq!(reply, "Echoed your request.");
        assert_eq!(agent.memory().len(), 4);
    }
}

