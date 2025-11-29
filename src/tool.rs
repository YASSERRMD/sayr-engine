use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{AgnoError, Result};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;

    /// Optionally return a JSON Schema-like object describing expected arguments.
    fn parameters(&self) -> Option<Value> {
        None
    }
    async fn call(&self, input: Value) -> Result<Value>;
}

/// Static description of a tool that can be embedded in prompts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolDescription {
    pub name: String,
    pub description: String,
    pub parameters: Option<Value>,
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
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn describe(&self) -> Vec<ToolDescription> {
        let mut descriptions: Vec<ToolDescription> = self
            .tools
            .values()
            .map(|tool| ToolDescription {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters(),
            })
            .collect();

        descriptions.sort_by(|a, b| a.name.cmp(&b.name));
        descriptions
    }

    pub async fn call(&self, name: &str, input: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| AgnoError::ToolNotFound(name.to_string()))?;
        tool.call(input)
            .await
            .map_err(|source| AgnoError::ToolInvocation {
                name: name.to_string(),
                source: Box::new(source),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes whatever input is provided"
        }

        fn parameters(&self) -> Option<Value> {
            Some(serde_json::json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }))
        }

        async fn call(&self, input: Value) -> Result<Value> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn describes_registered_tools_with_parameters() {
        let mut registry = ToolRegistry::new();
        registry.register(Echo);

        let descriptions = registry.describe();
        assert_eq!(descriptions.len(), 1);
        let desc = &descriptions[0];
        assert_eq!(desc.name, "echo");
        assert!(desc
            .parameters
            .as_ref()
            .unwrap()
            .get("properties")
            .is_some());

        let output = registry
            .call("echo", serde_json::json!({"text":"hi"}))
            .await
            .unwrap();
        assert_eq!(output["text"], "hi");
    }

    #[tokio::test]
    async fn describes_tools_in_deterministic_order() {
        struct Second;

        #[async_trait]
        impl Tool for Second {
            fn name(&self) -> &str {
                "second"
            }

            fn description(&self) -> &str {
                "Second tool"
            }

            async fn call(&self, input: Value) -> Result<Value> {
                Ok(input)
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Echo);
        registry.register(Second);

        let descriptions = registry.describe();
        let names: Vec<String> = descriptions.into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["echo", "second"]);
    }
}
