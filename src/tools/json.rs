//! JSON toolkit.
//!
//! Provides tools for parsing, validating, and manipulating JSON data.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// Create a JSON toolkit
pub fn json_toolkit() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(JsonParseTool);
    registry.register(JsonQueryTool);
    registry.register(JsonValidateTool);
    registry
}

struct JsonParseTool;

#[async_trait]
impl Tool for JsonParseTool {
    fn name(&self) -> &str {
        "json_parse"
    }

    fn description(&self) -> &str {
        "Parse a JSON string into structured data. Expects {\"text\": string}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let text = input
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `text` for json_parse".into()))?;

        match serde_json::from_str::<Value>(text) {
            Ok(parsed) => Ok(json!({
                "success": true,
                "data": parsed
            })),
            Err(e) => Ok(json!({
                "success": false,
                "error": e.to_string()
            })),
        }
    }
}

struct JsonQueryTool;

#[async_trait]
impl Tool for JsonQueryTool {
    fn name(&self) -> &str {
        "json_query"
    }

    fn description(&self) -> &str {
        "Query a JSON object using a dot-separated path. Expects {\"data\": object, \"path\": string}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let data = input
            .get("data")
            .ok_or_else(|| AgnoError::Protocol("missing `data` for json_query".into()))?;

        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `path` for json_query".into()))?;

        let result = query_json_path(data, path);

        Ok(json!({
            "path": path,
            "result": result,
            "found": !result.is_null()
        }))
    }
}

struct JsonValidateTool;

#[async_trait]
impl Tool for JsonValidateTool {
    fn name(&self) -> &str {
        "json_validate"
    }

    fn description(&self) -> &str {
        "Validate that a string is valid JSON. Expects {\"text\": string}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let text = input
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `text` for json_validate".into()))?;

        match serde_json::from_str::<Value>(text) {
            Ok(_) => Ok(json!({
                "valid": true
            })),
            Err(e) => Ok(json!({
                "valid": false,
                "error": e.to_string(),
                "line": e.line(),
                "column": e.column()
            })),
        }
    }
}

fn query_json_path(data: &Value, path: &str) -> Value {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = data;

    for part in parts {
        if part.is_empty() {
            continue;
        }

        // Try as array index
        if let Ok(index) = part.parse::<usize>() {
            if let Some(arr) = current.as_array() {
                if let Some(elem) = arr.get(index) {
                    current = elem;
                    continue;
                }
            }
        }

        // Try as object key
        if let Some(obj) = current.as_object() {
            if let Some(value) = obj.get(part) {
                current = value;
                continue;
            }
        }

        return Value::Null;
    }

    current.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_parse() {
        let registry = json_toolkit();
        let parse = registry.get("json_parse").unwrap();
        
        let result = parse.call(json!({"text": "{\"name\": \"test\"}"})).await.unwrap();
        assert_eq!(result["success"], true);
        assert_eq!(result["data"]["name"], "test");
    }

    #[tokio::test]
    async fn test_json_query() {
        let registry = json_toolkit();
        let query = registry.get("json_query").unwrap();
        
        let data = json!({
            "user": {
                "name": "Alice",
                "items": [1, 2, 3]
            }
        });
        
        let result = query.call(json!({
            "data": data,
            "path": "user.name"
        })).await.unwrap();
        
        assert_eq!(result["result"], "Alice");
        assert_eq!(result["found"], true);
    }

    #[tokio::test]
    async fn test_json_validate() {
        let registry = json_toolkit();
        let validate = registry.get("json_validate").unwrap();
        
        let result = validate.call(json!({"text": "{\"valid\": true}"})).await.unwrap();
        assert_eq!(result["valid"], true);
        
        let result = validate.call(json!({"text": "{invalid}"})).await.unwrap();
        assert_eq!(result["valid"], false);
    }
}
