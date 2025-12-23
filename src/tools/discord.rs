//! Discord toolkit for interacting with Discord servers.
//!
//! Provides tools for sending messages and listing channels via Discord bot API.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Discord Client
// ─────────────────────────────────────────────────────────────────────────────

/// Discord API client
#[derive(Clone)]
pub struct DiscordClient {
    http: reqwest::Client,
    bot_token: String,
    base_url: String,
}

impl DiscordClient {
    pub fn new(bot_token: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            bot_token: bot_token.into(),
            base_url: "https://discord.com/api/v10".to_string(),
        }
    }

    pub fn from_env() -> crate::Result<Self> {
        let token = std::env::var("DISCORD_BOT_TOKEN")
            .map_err(|_| crate::error::AgnoError::Protocol("DISCORD_BOT_TOKEN not set".into()))?;
        Ok(Self::new(token))
    }

    async fn get(&self, endpoint: &str) -> crate::Result<Value> {
        let response = self
            .http
            .get(format!("{}{}", self.base_url, endpoint))
            .header("Authorization", format!("Bot {}", self.bot_token))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Discord request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::AgnoError::Protocol(format!(
                "Discord API error {}: {}",
                status, body
            )));
        }

        response
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse response: {}", e)))
    }

    async fn post(&self, endpoint: &str, body: Value) -> crate::Result<Value> {
        let response = self
            .http
            .post(format!("{}{}", self.base_url, endpoint))
            .header("Authorization", format!("Bot {}", self.bot_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Discord request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::AgnoError::Protocol(format!(
                "Discord API error {}: {}",
                status, body
            )));
        }

        response
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse response: {}", e)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Message Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for sending messages to Discord channels
pub struct DiscordSendMessageTool {
    client: DiscordClient,
}

impl DiscordSendMessageTool {
    pub fn new(client: DiscordClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(DiscordClient::from_env()?))
    }
}

#[async_trait]
impl Tool for DiscordSendMessageTool {
    fn name(&self) -> &str {
        "discord_send_message"
    }

    fn description(&self) -> &str {
        "Send a message to a Discord channel."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The Discord channel ID"
                },
                "content": {
                    "type": "string",
                    "description": "Message content to send"
                }
            },
            "required": ["channel_id", "content"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let channel_id = input["channel_id"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'channel_id' parameter".into()))?;
        let content = input["content"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'content' parameter".into()))?;

        let response = self.client.post(
            &format!("/channels/{}/messages", channel_id),
            json!({ "content": content })
        ).await?;

        Ok(json!({
            "success": true,
            "message_id": response["id"],
            "channel_id": response["channel_id"],
            "content": response["content"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// List Guild Channels Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for listing channels in a Discord guild
pub struct DiscordListChannelsTool {
    client: DiscordClient,
}

impl DiscordListChannelsTool {
    pub fn new(client: DiscordClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(DiscordClient::from_env()?))
    }
}

#[async_trait]
impl Tool for DiscordListChannelsTool {
    fn name(&self) -> &str {
        "discord_list_channels"
    }

    fn description(&self) -> &str {
        "List all channels in a Discord guild (server)."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "The Discord guild (server) ID"
                }
            },
            "required": ["guild_id"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let guild_id = input["guild_id"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'guild_id' parameter".into()))?;

        let response = self.client.get(&format!("/guilds/{}/channels", guild_id)).await?;

        let channels = response
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|ch| {
                        let channel_type = match ch["type"].as_i64() {
                            Some(0) => "text",
                            Some(2) => "voice",
                            Some(4) => "category",
                            Some(5) => "announcement",
                            Some(13) => "stage",
                            Some(15) => "forum",
                            _ => "other",
                        };
                        json!({
                            "id": ch["id"],
                            "name": ch["name"],
                            "type": channel_type,
                            "position": ch["position"],
                            "parent_id": ch["parent_id"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "guild_id": guild_id,
            "channels": channels,
            "count": channels.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Get Channel Messages Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for getting messages from a Discord channel
pub struct DiscordGetMessagesTool {
    client: DiscordClient,
}

impl DiscordGetMessagesTool {
    pub fn new(client: DiscordClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(DiscordClient::from_env()?))
    }
}

#[async_trait]
impl Tool for DiscordGetMessagesTool {
    fn name(&self) -> &str {
        "discord_get_messages"
    }

    fn description(&self) -> &str {
        "Get recent messages from a Discord channel."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The Discord channel ID"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of messages to retrieve (max 100, default 50)"
                }
            },
            "required": ["channel_id"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let channel_id = input["channel_id"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'channel_id' parameter".into()))?;
        let limit = input["limit"].as_i64().unwrap_or(50).min(100);

        let response = self.client.get(&format!("/channels/{}/messages?limit={}", channel_id, limit)).await?;

        let messages = response
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|msg| {
                        json!({
                            "id": msg["id"],
                            "content": msg["content"],
                            "author": {
                                "id": msg["author"]["id"],
                                "username": msg["author"]["username"]
                            },
                            "timestamp": msg["timestamp"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "channel_id": channel_id,
            "messages": messages,
            "count": messages.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Discord Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all Discord tools with a registry
pub fn register_discord_tools(registry: &mut ToolRegistry, bot_token: impl Into<String>) {
    let client = DiscordClient::new(bot_token);
    registry.register(DiscordSendMessageTool::new(client.clone()));
    registry.register(DiscordListChannelsTool::new(client.clone()));
    registry.register(DiscordGetMessagesTool::new(client));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discord_client_creation() {
        let client = DiscordClient::new("test-token");
        assert_eq!(client.bot_token, "test-token");
    }

    #[test]
    fn test_discord_tools_creation() {
        let client = DiscordClient::new("test");
        let send = DiscordSendMessageTool::new(client.clone());
        assert_eq!(send.name(), "discord_send_message");
        
        let list = DiscordListChannelsTool::new(client.clone());
        assert_eq!(list.name(), "discord_list_channels");
        
        let get = DiscordGetMessagesTool::new(client);
        assert_eq!(get.name(), "discord_get_messages");
    }
}
