use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::agent::Agent;
use crate::memory::ConversationMemory;
use crate::{LanguageModel, Result};

/// Events emitted by the team bus.
#[derive(Debug, Clone)]
pub enum TeamEvent {
    Broadcast { from: String, content: String },
    KnowledgeAdded(String),
}

/// A coordination surface for multiple agents that share context and a message bus.
pub struct Team<M: LanguageModel> {
    name: String,
    members: BTreeMap<String, Arc<Mutex<Agent<M>>>>,
    shared_memory: Arc<RwLock<ConversationMemory>>,
    shared_context: Arc<RwLock<Value>>,
    knowledge: Arc<RwLock<Vec<String>>>,
    tx: broadcast::Sender<TeamEvent>,
}

impl<M: LanguageModel> Clone for Team<M> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            members: self.members.clone(),
            shared_memory: Arc::clone(&self.shared_memory),
            shared_context: Arc::clone(&self.shared_context),
            knowledge: Arc::clone(&self.knowledge),
            tx: self.tx.clone(),
        }
    }
}

impl<M: LanguageModel> Team<M> {
    /// Create an empty team with a broadcast bus and shared memory.
    pub fn new(name: impl Into<String>) -> Self {
        let (tx, _) = broadcast::channel(128);
        Self {
            name: name.into(),
            members: BTreeMap::new(),
            shared_memory: Arc::new(RwLock::new(ConversationMemory::default())),
            shared_context: Arc::new(RwLock::new(Value::Null)),
            knowledge: Arc::new(RwLock::new(Vec::new())),
            tx,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Register a new agent under the given identifier.
    pub fn add_agent(&mut self, id: impl Into<String>, agent: Agent<M>) {
        self.members.insert(id.into(), Arc::new(Mutex::new(agent)));
    }

    /// Number of registered agents.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Subscribe to the broadcast bus for inter-agent notifications.
    pub fn subscribe(&self) -> broadcast::Receiver<TeamEvent> {
        self.tx.subscribe()
    }

    /// Append shared knowledge that all agents can reference.
    pub async fn add_knowledge(&self, fact: impl Into<String>) {
        let fact = fact.into();
        self.knowledge.write().await.push(fact.clone());
        let _ = self.tx.send(TeamEvent::KnowledgeAdded(fact));
    }

    /// Update the shared context blob (typically JSON state shared across steps).
    pub async fn set_context(&self, ctx: Value) {
        *self.shared_context.write().await = ctx;
    }

    /// Retrieve a copy of the shared context.
    pub async fn context(&self) -> Value {
        self.shared_context.read().await.clone()
    }

    /// Send a broadcast message to all listeners and append to shared memory.
    pub async fn broadcast(&self, from: impl Into<String>, content: impl Into<String>) {
        let from = from.into();
        let content = content.into();
        if let Ok(mut memory) = self.shared_memory.try_write() {
            memory.push(crate::message::Message::assistant(format!(
                "[{from}] {content}"
            )));
        }
        let _ = self.tx.send(TeamEvent::Broadcast { from, content });
    }

    /// Run the same prompt through every agent, synchronizing memory back into the shared
    /// transcript after each response. Returns agent replies in registration order.
    pub async fn fan_out(&self, prompt: &str) -> Result<Vec<(String, String)>> {
        let mut replies = Vec::new();
        for (id, agent) in &self.members {
            let mut guard = agent.lock().await;
            // Share the latest transcript with the agent before it responds.
            let snapshot = { self.shared_memory.read().await.clone() };
            guard.sync_memory_from(&snapshot);
            let reply = guard.respond(prompt).await?;
            // Persist the updated transcript back into the shared memory.
            let updated = guard.take_memory_snapshot();
            *self.shared_memory.write().await = updated;
            replies.push((id.clone(), reply));
        }
        Ok(replies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Agent, StubModel};

    #[tokio::test]
    async fn runs_agents_with_shared_memory() {
        let a_model = StubModel::new(vec![
            r#"{"action":"respond","content":"a2"}"#.into(),
            r#"{"action":"respond","content":"a1"}"#.into(),
        ]);
        let b_model = StubModel::new(vec![
            r#"{"action":"respond","content":"b2"}"#.into(),
            r#"{"action":"respond","content":"b1"}"#.into(),
        ]);

        let mut team = Team::new("demo");
        team.add_agent("alpha", Agent::new(a_model));
        team.add_agent("beta", Agent::new(b_model));

        let replies = team.fan_out("hello world").await.unwrap();
        assert_eq!(replies.len(), 2);
        assert_eq!(replies[0].1, "a1");
        assert_eq!(replies[1].1, "b1");

        // second pass reads and writes the shared transcript again
        let replies = team.fan_out("follow up").await.unwrap();
        assert_eq!(replies[0].1, "a2");
        assert_eq!(replies[1].1, "b2");
    }
}
