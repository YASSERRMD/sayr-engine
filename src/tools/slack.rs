//! Slack toolkit for interacting with Slack workspaces.
//!
//! Provides tools for sending messages, listing channels, and searching messages.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Slack Client
// ─────────────────────────────────────────────────────────────────────────────

/// Shared Slack API client
#[derive(Clone)]
pub struct SlackClient {
    http: reqwest::Client,
    token: String,
    base_url: String,
}

impl SlackClient {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            token: token.into(),
            base_url: "https://slack.com/api".to_string(),
        }
    }

    pub fn from_env() -> crate::Result<Self> {
        let token = std::env::var("SLACK_BOT_TOKEN")
            .or_else(|_| std::env::var("SLACK_TOKEN"))
            .map_err(|_| crate::error::AgnoError::Protocol("SLACK_BOT_TOKEN not set".into()))?;
        Ok(Self::new(token))
    }

    async fn post(&self, endpoint: &str, body: Value) -> crate::Result<Value> {
        let response = self
            .http
            .post(format!("{}/{}", self.base_url, endpoint))
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Slack request failed: {}", e)))?;

        let result: Value = response
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse response: {}", e)))?;

        if result["ok"].as_bool() != Some(true) {
            let error = result["error"].as_str().unwrap_or("unknown error");
            return Err(crate::error::AgnoError::Protocol(format!(
                "Slack API error: {}",
                error
            )));
        }

        Ok(result)
    }

    async fn get(&self, endpoint: &str) -> crate::Result<Value> {
        let response = self
            .http
            .get(format!("{}/{}", self.base_url, endpoint))
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Slack request failed: {}", e)))?;

        let result: Value = response
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse response: {}", e)))?;

        if result["ok"].as_bool() != Some(true) {
            let error = result["error"].as_str().unwrap_or("unknown error");
            return Err(crate::error::AgnoError::Protocol(format!(
                "Slack API error: {}",
                error
            )));
        }

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Message Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for sending messages to Slack channels
pub struct SlackSendMessageTool {
    client: SlackClient,
}

impl SlackSendMessageTool {
    pub fn new(client: SlackClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(SlackClient::from_env()?))
    }
}

#[async_trait]
impl Tool for SlackSendMessageTool {
    fn name(&self) -> &str {
        "slack_send_message"
    }

    fn description(&self) -> &str {
        "Send a message to a Slack channel or user."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel ID or name (e.g., #general or C1234567890)"
                },
                "text": {
                    "type": "string",
                    "description": "Message text to send"
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Optional thread timestamp to reply to"
                }
            },
            "required": ["channel", "text"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let channel = input["channel"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'channel' parameter".into()))?;
        let text = input["text"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'text' parameter".into()))?;

        let mut body = json!({
            "channel": channel,
            "text": text
        });

        if let Some(thread_ts) = input["thread_ts"].as_str() {
            body["thread_ts"] = json!(thread_ts);
        }

        let response = self.client.post("chat.postMessage", body).await?;

        Ok(json!({
            "success": true,
            "channel": response["channel"],
            "ts": response["ts"],
            "message": response["message"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// List Channels Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for listing Slack channels
pub struct SlackListChannelsTool {
    client: SlackClient,
}

impl SlackListChannelsTool {
    pub fn new(client: SlackClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(SlackClient::from_env()?))
    }
}

#[async_trait]
impl Tool for SlackListChannelsTool {
    fn name(&self) -> &str {
        "slack_list_channels"
    }

    fn description(&self) -> &str {
        "List available Slack channels in the workspace."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "types": {
                    "type": "string",
                    "description": "Channel types: public_channel, private_channel, mpim, im"
                }
            }
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let types = input["types"].as_str().unwrap_or("public_channel");
        
        let endpoint = format!("conversations.list?types={}&limit=100", types);
        let response = self.client.get(&endpoint).await?;

        let channels = response["channels"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|ch| {
                        json!({
                            "id": ch["id"],
                            "name": ch["name"],
                            "is_private": ch["is_private"],
                            "num_members": ch["num_members"],
                            "topic": ch["topic"]["value"],
                            "purpose": ch["purpose"]["value"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "channels": channels,
            "count": channels.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search Messages Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for searching Slack messages
pub struct SlackSearchTool {
    client: SlackClient,
}

impl SlackSearchTool {
    pub fn new(client: SlackClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(SlackClient::from_env()?))
    }
}

#[async_trait]
impl Tool for SlackSearchTool {
    fn name(&self) -> &str {
        "slack_search"
    }

    fn description(&self) -> &str {
        "Search for messages in Slack."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return (default: 20)"
                }
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'query' parameter".into()))?;
        let count = input["count"].as_i64().unwrap_or(20);

        let endpoint = format!(
            "search.messages?query={}&count={}",
            urlencoding::encode(query),
            count
        );
        let response = self.client.get(&endpoint).await?;

        let messages = response["messages"]["matches"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|msg| {
                        json!({
                            "text": msg["text"],
                            "user": msg["user"],
                            "channel": msg["channel"]["name"],
                            "ts": msg["ts"],
                            "permalink": msg["permalink"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "query": query,
            "messages": messages,
            "total": response["messages"]["total"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slack Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all Slack tools with a registry
pub fn register_slack_tools(registry: &mut ToolRegistry, token: impl Into<String>) {
    let client = SlackClient::new(token);
    registry.register(SlackSendMessageTool::new(client.clone()));
    registry.register(SlackListChannelsTool::new(client.clone()));
    registry.register(SlackSearchTool::new(client));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slack_client_creation() {
        let client = SlackClient::new("test-token");
        assert_eq!(client.token, "test-token");
    }
}
