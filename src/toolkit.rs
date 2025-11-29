use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use tokio::{fs, io::AsyncWriteExt};

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

pub fn basic_toolkit() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool);
    registry.register(WriteFileTool);
    registry.register(EchoTool);
    registry
}

struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a UTF-8 file. Expects {\"path\": string}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `path` for read_file".into()))?;

        let contents = fs::read_to_string(path)
            .await
            .map_err(|err| AgnoError::ToolInvocation {
                name: self.name().into(),
                source: Box::new(err),
            })?;

        Ok(json!({ "path": path, "contents": contents }))
    }
}

struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write contents to a UTF-8 file. Expects {\"path\": string, \"contents\": string}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `path` for write_file".into()))?;
        let contents = input
            .get("contents")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `contents` for write_file".into()))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .await
            .map_err(|err| AgnoError::ToolInvocation {
                name: self.name().into(),
                source: Box::new(err),
            })?;

        file.write_all(contents.as_bytes())
            .await
            .map_err(|err| AgnoError::ToolInvocation {
                name: self.name().into(),
                source: Box::new(err),
            })?;

        Ok(json!({ "path": path, "bytes_written": contents.len() }))
    }
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echo the provided JSON payload back to the caller."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        Ok(json!({ "echo": input }))
    }
}
