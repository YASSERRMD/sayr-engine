use async_trait::async_trait;

use crate::error::Result;
use crate::message::{Message, ToolCall, ToolResult};

#[async_trait]
pub trait AgentHook: Send + Sync {
    async fn before_model(&self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }

    async fn after_model(&self, _raw_response: &str) -> Result<()> {
        Ok(())
    }

    async fn before_tool_call(&self, _call: &ToolCall) -> Result<()> {
        Ok(())
    }

    async fn after_tool_result(&self, _result: &ToolResult) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
pub trait ConfirmationHandler: Send + Sync {
    async fn confirm_tool_call(&self, call: &ToolCall) -> Result<bool>;
}
