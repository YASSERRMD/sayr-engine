use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::error::{AgnoError, Result};

/// Minimal abstraction around a chat completion provider.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
}

/// A deterministic model used for tests and demos.
pub struct StubModel {
    responses: Mutex<Vec<String>>,
}

impl StubModel {
    pub fn new(responses: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses),
        })
    }
}

#[async_trait]
impl LanguageModel for StubModel {
    async fn complete(&self, _prompt: &str) -> Result<String> {
        let mut locked = self.responses.lock().expect("stub model poisoned");
        locked.pop().ok_or_else(|| {
            AgnoError::LanguageModel("StubModel ran out of scripted responses".into())
        })
    }
}
