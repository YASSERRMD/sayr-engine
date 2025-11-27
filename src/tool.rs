use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{AgnoError, Result};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn call(&self, input: Value) -> Result<Value>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>, 
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools
            .insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub async fn call(&self, name: &str, input: Value) -> Result<Value> {
        let tool = self.tools.get(name).ok_or_else(|| AgnoError::ToolNotFound(name.to_string()))?;
        tool.call(input)
            .await
            .map_err(|source| AgnoError::ToolInvocation { name: name.to_string(), source })
    }
}

