use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{AgnoError, Result};
use crate::governance::{AccessController, Action, Principal, Role as GovernanceRole};
use crate::hooks::{AgentHook, ConfirmationHandler};
use crate::knowledge::Retriever;
use crate::llm::LanguageModel;
use crate::memory::ConversationMemory;
use crate::message::{Message, Role, ToolCall};
use crate::metrics::{MetricsTracker, RunGuard};
use crate::telemetry::{TelemetryCollector, TelemetryLabels};
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
    input_schema: Option<serde_json::Value>,
    output_schema: Option<serde_json::Value>,
    hooks: Vec<Arc<dyn AgentHook>>,
    retriever: Option<Arc<dyn Retriever>>,
    require_tool_confirmation: bool,
    confirmation_handler: Option<Arc<dyn ConfirmationHandler>>,
    access_control: Option<Arc<AccessController>>,
    principal: Principal,
    metrics: Option<MetricsTracker>,
    telemetry: Option<TelemetryCollector>,
    workflow_label: Option<String>,
}

impl<M: LanguageModel> Agent<M> {
    pub fn new(model: Arc<M>) -> Self {
        Self {
            system_prompt: "You are a helpful agent.".to_string(),
            model,
            tools: ToolRegistry::new(),
            memory: ConversationMemory::default(),
            max_steps: 6,
            input_schema: None,
            output_schema: None,
            hooks: Vec::new(),
            retriever: None,
            require_tool_confirmation: false,
            confirmation_handler: None,
            access_control: None,
            principal: Principal {
                id: "anonymous".into(),
                role: GovernanceRole::User,
                tenant: None,
            },
            metrics: None,
            telemetry: None,
            workflow_label: None,
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

    pub fn with_access_control(mut self, controller: Arc<AccessController>) -> Self {
        self.access_control = Some(controller);
        self
    }

    pub fn with_principal(mut self, principal: Principal) -> Self {
        self.principal = principal;
        self
    }

    pub fn with_metrics(mut self, metrics: MetricsTracker) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_telemetry(mut self, telemetry: TelemetryCollector) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    pub fn with_workflow_label(mut self, workflow: impl Into<String>) -> Self {
        self.workflow_label = Some(workflow.into());
        self
    }

    pub fn with_input_schema(mut self, schema: serde_json::Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    pub fn with_output_schema(mut self, schema: serde_json::Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn with_hook(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    pub fn with_retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    pub fn require_tool_confirmation(mut self, handler: Arc<dyn ConfirmationHandler>) -> Self {
        self.require_tool_confirmation = true;
        self.confirmation_handler = Some(handler);
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

    pub fn sync_memory_from(&mut self, memory: &ConversationMemory) {
        self.memory = memory.clone();
    }

    pub fn take_memory_snapshot(&self) -> ConversationMemory {
        self.memory.clone()
    }

    /// Run a single exchange with the agent. Returns the final assistant reply.
    pub async fn respond(&mut self, user_input: impl Into<String>) -> Result<String> {
        let principal = self.principal.clone();
        self.respond_for(principal, user_input).await
    }

    pub async fn respond_for(
        &mut self,
        principal: Principal,
        user_input: impl Into<String>,
    ) -> Result<String> {
        if let Some(ctrl) = &self.access_control {
            if !ctrl.authorize(&principal, &Action::SendMessage) {
                return Err(AgnoError::Protocol(
                    "principal not authorized to send messages".into(),
                ));
            }
        }

        let base_labels = TelemetryLabels {
            tenant: principal.tenant.clone(),
            tool: None,
            workflow: self.workflow_label.clone(),
        };
        if let Some(telemetry) = &self.telemetry {
            telemetry.record(
                "user_message",
                serde_json::json!({"principal": principal.id.clone(), "tenant": principal.tenant}),
                base_labels.clone(),
            );
        }

        let mut run_guard: Option<RunGuard> = self
            .metrics
            .as_ref()
            .map(|m| m.start_run(base_labels.clone()));
        self.memory.push(Message::user(user_input));

        for _ in 0..self.max_steps {
            let prompt = self.build_prompt().await?;
            let snapshot: Vec<Message> = self.memory.iter().cloned().collect();
            for hook in &self.hooks {
                hook.before_model(snapshot.as_slice()).await?;
            }
            let raw = self.model.complete(&prompt).await?;
            for hook in &self.hooks {
                hook.after_model(&raw).await?;
            }
            let directive: AgentDirective = serde_json::from_str(&raw).map_err(|err| {
                AgnoError::Protocol(format!(
                    "Expected JSON directive with `action`, got `{raw}`: {err}"
                ))
            })?;

            match directive {
                AgentDirective::Respond { content } => {
                    self.memory.push(Message::assistant(&content));
                    if let Some(guard) = run_guard.take() {
                        guard.finish(true);
                    }
                    return Ok(content);
                }
                AgentDirective::CallTool { name, arguments } => {
                    if let Some(ctrl) = &self.access_control {
                        if !ctrl.authorize(&principal, &Action::CallTool(name.clone())) {
                            if let Some(guard) = run_guard.as_mut() {
                                guard.record_failure(Some(name.clone()));
                            }
                            return Err(AgnoError::Protocol(format!(
                                "principal `{}` not allowed to call tool `{}`",
                                principal.id, name
                            )));
                        }
                    }
                    if self.require_tool_confirmation {
                        if let Some(handler) = &self.confirmation_handler {
                            let approved = handler
                                .confirm_tool_call(&ToolCall {
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                })
                                .await?;
                            if !approved {
                                self.memory.push(Message::assistant(format!(
                                    "Tool call `{name}` rejected by guardrail",
                                )));
                                continue;
                            }
                        }
                    }
                    if let Some(guard) = run_guard.as_mut() {
                        guard.record_tool_call(name.clone());
                    }
                    self.memory.push(Message {
                        role: Role::Assistant,
                        content: format!("Calling tool `{name}`"),
                        tool_call: Some(ToolCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        }),
                        tool_result: None,
                        attachments: Vec::new(),
                    });

                    for hook in &self.hooks {
                        hook.before_tool_call(
                            self.memory
                                .iter()
                                .last()
                                .unwrap()
                                .tool_call
                                .as_ref()
                                .unwrap(),
                        )
                        .await?;
                    }
                    let output = match self.tools.call(&name, arguments.clone()).await {
                        Ok(value) => value,
                        Err(err) => {
                            if let Some(guard) = run_guard.as_mut() {
                                guard.record_failure(Some(name.clone()));
                            }
                            if let Some(telemetry) = &self.telemetry {
                                telemetry.record_failure(
                                    format!("tool::{name}"),
                                    format!("{err}"),
                                    0,
                                    base_labels.clone().with_tool(name.clone()),
                                );
                            }
                            return Err(err);
                        }
                    };
                    let result_message = Message::tool(&name, output);
                    for hook in &self.hooks {
                        if let Some(result) = result_message.tool_result.as_ref() {
                            hook.after_tool_result(result).await?;
                        }
                    }
                    self.memory.push(result_message);
                }
            }
        }

        if let Some(guard) = run_guard {
            guard.finish(false);
        }

        Err(AgnoError::Protocol(
            "Agent reached the step limit without returning a response".into(),
        ))
    }

    async fn build_prompt(&self) -> Result<String> {
        let mut prompt = String::new();
        prompt.push_str(&format!("System: {}\n\n", self.system_prompt));
        if let Some(schema) = &self.input_schema {
            prompt.push_str(&format!(
                "User input is expected to follow this JSON shape: {}\n\n",
                schema
            ));
        }
        prompt.push_str("You must answer with JSON in one of the following formats:\n");
        prompt.push_str("- {\"action\":\"respond\",\"content\":\"<final assistant message>\"}\n");
        prompt.push_str(
            "- {\"action\":\"call_tool\",\"name\":\"<tool name>\",\"arguments\":{...}}\n\n",
        );
        if let Some(schema) = &self.output_schema {
            prompt.push_str(&format!(
                "When responding directly, conform to this output schema: {}\n\n",
                schema
            ));
        }
        if self.tools.names().is_empty() {
            prompt.push_str("No tools are available.\n\n");
        } else {
            prompt.push_str("Available tools:\n");
            for tool in self.tools.describe() {
                prompt.push_str(&format!("- {}: {}", tool.name, tool.description));
                if let Some(params) = &tool.parameters {
                    prompt.push_str(&format!(" (parameters: {})", params));
                }
                prompt.push('\n');
            }
            prompt.push('\n');
        }

        if let Some(retriever) = &self.retriever {
            let contexts = retriever
                .retrieve(
                    self.memory
                        .iter()
                        .rev()
                        .find(|m| m.role == Role::User)
                        .map(|m| m.content.as_str())
                        .unwrap_or_default(),
                    3,
                )
                .await
                .unwrap_or_default();
            if !contexts.is_empty() {
                prompt.push_str("Context snippets:\n");
                for ctx in contexts {
                    prompt.push_str("- ");
                    prompt.push_str(&ctx);
                    prompt.push('\n');
                }
                prompt.push('\n');
            }
        }

        prompt.push_str("Conversation so far:\n");
        for message in self.memory.iter() {
            prompt.push_str(&format!("[{:?}] {}\n", message.role, message.content));
            if let Some(call) = &message.tool_call {
                prompt.push_str(&format!(
                    "  -> calling {} with {}\n",
                    call.name, call.arguments
                ));
            }
            if let Some(result) = &message.tool_result {
                prompt.push_str(&format!(
                    "  <- {} returned {}\n",
                    result.name, result.output
                ));
            }
        }

        Ok(prompt)
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

    #[tokio::test]
    async fn includes_tool_metadata_in_prompt() {
        struct DescribingTool;

        #[async_trait]
        impl Tool for DescribingTool {
            fn name(&self) -> &str {
                "describe"
            }

            fn description(&self) -> &str {
                "Replies with metadata"
            }

            fn parameters(&self) -> Option<Value> {
                Some(serde_json::json!({"type":"object","properties":{"id":{"type":"string"}}}))
            }

            async fn call(&self, _input: Value) -> Result<Value> {
                Ok(serde_json::json!({"ok": true}))
            }
        }

        let model = StubModel::new(vec![r#"{"action":"respond","content":"done"}"#.into()]);
        let mut tools = ToolRegistry::new();
        tools.register(DescribingTool);

        let agent = Agent::new(model).with_tools(tools);
        let prompt = agent.build_prompt().await.unwrap();

        assert!(prompt.contains("describe: Replies with metadata"));
        assert!(prompt.contains("parameters"));
    }
}
