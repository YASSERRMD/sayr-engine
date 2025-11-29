use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{broadcast, RwLock};

use crate::agent::Agent;
use crate::error::Result;
use crate::llm::LanguageModel;
use crate::memory::ConversationMemory;
use crate::message::{Message, Role};

/// A lightweight message exchanged between team members.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeamMessage {
    pub from: String,
    pub to: Option<String>,
    pub content: String,
}

/// Shared context that can be read/written by all agents in the team.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedContext {
    pub memory: ConversationMemory,
    pub state: serde_json::Value,
}

impl Default for SharedContext {
    fn default() -> Self {
        Self {
            memory: ConversationMemory::default(),
            state: json!({}),
        }
    }
}

/// A collection of collaborating agents that share state and communicate via a bus.
pub struct Team<M: LanguageModel> {
    name: String,
    members: HashMap<String, Agent<M>>,
    context: Arc<RwLock<SharedContext>>,
    mailbox: broadcast::Sender<TeamMessage>,
}

impl<M: LanguageModel> Team<M> {
    pub fn new(name: impl Into<String>) -> Self {
        let (mailbox, _) = broadcast::channel(64);
        Self {
            name: name.into(),
            members: HashMap::new(),
            context: Arc::new(RwLock::new(SharedContext::default())),
            mailbox,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn context(&self) -> Arc<RwLock<SharedContext>> {
        Arc::clone(&self.context)
    }

    pub fn add_member(&mut self, name: impl Into<String>, agent: Agent<M>) {
        self.members.insert(name.into(), agent);
    }

    pub fn member_names(&self) -> Vec<String> {
        self.members.keys().cloned().collect()
    }

    /// Replace or insert a state value within the shared team context.
    pub async fn set_state(&self, key: impl Into<String>, value: serde_json::Value) {
        let mut ctx = self.context.write().await;
        if !ctx.state.is_object() {
            ctx.state = json!({});
        }
        if let Some(map) = ctx.state.as_object_mut() {
            map.insert(key.into(), value);
        }
    }

    /// Retrieve the full shared state as a clone for downstream consumers.
    pub async fn state(&self) -> serde_json::Value {
        self.context.read().await.state.clone()
    }

    /// Snapshot the shared conversation memory for telemetry or UI purposes.
    pub async fn memory(&self) -> ConversationMemory {
        self.context.read().await.memory.clone()
    }

    /// Broadcast a note to all team members and persist it in shared memory.
    pub async fn broadcast(&self, from: impl Into<String>, content: impl Into<String>) {
        let msg = TeamMessage {
            from: from.into(),
            to: None,
            content: content.into(),
        };
        let _ = self.mailbox.send(msg.clone());
        let mut ctx = self.context.write().await;
        ctx.memory.push(Message {
            role: Role::User,
            content: format!("[broadcast] {}", msg.content),
            tool_call: None,
            tool_result: None,
            attachments: Vec::new(),
        });
    }

    /// Send a direct message to a member. It will be added to the shared log so all
    /// agents can reason about past conversations.
    pub async fn send_message(
        &self,
        from: impl Into<String>,
        to: impl Into<String>,
        content: impl Into<String>,
    ) {
        let msg = TeamMessage {
            from: from.into(),
            to: Some(to.into()),
            content: content.into(),
        };
        let _ = self.mailbox.send(msg.clone());
        let mut ctx = self.context.write().await;
        ctx.memory.push(Message {
            role: Role::User,
            content: format!(
                "[dm:{}] {}",
                msg.to.clone().unwrap_or_default(),
                msg.content
            ),
            tool_call: None,
            tool_result: None,
            attachments: Vec::new(),
        });
    }

    /// Run a specific agent with access to the shared context and return the reply.
    pub async fn run_agent(
        &mut self,
        member: &str,
        user_input: impl Into<String>,
    ) -> Result<String> {
        let Some(agent) = self.members.get_mut(member) else {
            return Err(crate::error::AgnoError::Protocol(format!(
                "Unknown team member `{member}`"
            )));
        };

        // synchronize the shared transcript into the agent's local memory
        let ctx = self.context.read().await;
        for msg in ctx.memory.iter() {
            agent.memory_mut().push(msg.clone());
        }

        let reply = agent.respond(user_input).await?;

        // record the reply in the shared memory so subsequent steps can see it
        drop(ctx);
        let mut ctx = self.context.write().await;
        ctx.memory
            .push(Message::assistant(format!("[{member}] {reply}")));
        Ok(reply)
    }

    /// Subscribe to the live message bus for real-time dashboards.
    pub fn subscribe(&self) -> broadcast::Receiver<TeamMessage> {
        self.mailbox.subscribe()
    }
}
