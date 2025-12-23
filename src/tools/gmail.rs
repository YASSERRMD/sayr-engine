//! Gmail toolkit for reading and sending emails.
//!
//! Provides tools for interacting with Gmail via the Google API.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Gmail Client
// ─────────────────────────────────────────────────────────────────────────────

/// Gmail API client
#[derive(Clone)]
pub struct GmailClient {
    http: reqwest::Client,
    access_token: String,
    base_url: String,
}

impl GmailClient {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            access_token: access_token.into(),
            base_url: "https://gmail.googleapis.com/gmail/v1".to_string(),
        }
    }

    pub fn from_env() -> crate::Result<Self> {
        let token = std::env::var("GMAIL_ACCESS_TOKEN")
            .or_else(|_| std::env::var("GOOGLE_ACCESS_TOKEN"))
            .map_err(|_| crate::error::AgnoError::Protocol("GMAIL_ACCESS_TOKEN not set".into()))?;
        Ok(Self::new(token))
    }

    async fn get(&self, endpoint: &str) -> crate::Result<Value> {
        let response = self
            .http
            .get(format!("{}{}", self.base_url, endpoint))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Gmail request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::AgnoError::Protocol(format!(
                "Gmail API error {}: {}",
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
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Gmail request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::AgnoError::Protocol(format!(
                "Gmail API error {}: {}",
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
// List Messages Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for listing Gmail messages
pub struct GmailListMessagesTool {
    client: GmailClient,
}

impl GmailListMessagesTool {
    pub fn new(client: GmailClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(GmailClient::from_env()?))
    }
}

#[async_trait]
impl Tool for GmailListMessagesTool {
    fn name(&self) -> &str {
        "gmail_list_messages"
    }

    fn description(&self) -> &str {
        "List recent emails from Gmail inbox."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'from:example@gmail.com', 'is:unread')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 10)"
                }
            }
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"].as_str().unwrap_or("");
        let max_results = input["max_results"].as_i64().unwrap_or(10);

        let mut endpoint = format!("/users/me/messages?maxResults={}", max_results);
        if !query.is_empty() {
            endpoint.push_str(&format!("&q={}", urlencoding::encode(query)));
        }

        let response = self.client.get(&endpoint).await?;

        let message_ids: Vec<&str> = response["messages"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|m| m["id"].as_str()).collect())
            .unwrap_or_default();

        // Fetch details for each message
        let mut messages = Vec::new();
        for id in message_ids.iter().take(10) {
            if let Ok(msg) = self.client.get(&format!("/users/me/messages/{}", id)).await {
                let headers = msg["payload"]["headers"].as_array();
                let get_header = |name: &str| -> Option<String> {
                    headers.and_then(|h| {
                        h.iter()
                            .find(|x| x["name"].as_str() == Some(name))
                            .and_then(|x| x["value"].as_str())
                            .map(String::from)
                    })
                };

                messages.push(json!({
                    "id": id,
                    "thread_id": msg["threadId"],
                    "subject": get_header("Subject"),
                    "from": get_header("From"),
                    "to": get_header("To"),
                    "date": get_header("Date"),
                    "snippet": msg["snippet"]
                }));
            }
        }

        Ok(json!({
            "query": query,
            "messages": messages,
            "count": messages.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Read Message Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for reading a specific Gmail message
pub struct GmailReadMessageTool {
    client: GmailClient,
}

impl GmailReadMessageTool {
    pub fn new(client: GmailClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(GmailClient::from_env()?))
    }
}

#[async_trait]
impl Tool for GmailReadMessageTool {
    fn name(&self) -> &str {
        "gmail_read_message"
    }

    fn description(&self) -> &str {
        "Read the full content of a specific Gmail message."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The Gmail message ID"
                }
            },
            "required": ["message_id"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let message_id = input["message_id"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'message_id' parameter".into()))?;

        let response = self.client.get(&format!("/users/me/messages/{}", message_id)).await?;

        let headers = response["payload"]["headers"].as_array();
        let get_header = |name: &str| -> Option<String> {
            headers.and_then(|h| {
                h.iter()
                    .find(|x| x["name"].as_str() == Some(name))
                    .and_then(|x| x["value"].as_str())
                    .map(String::from)
            })
        };

        // Try to get the body content
        let body = response["payload"]["body"]["data"]
            .as_str()
            .or_else(|| {
                response["payload"]["parts"]
                    .as_array()
                    .and_then(|parts| {
                        parts.iter()
                            .find(|p| p["mimeType"].as_str() == Some("text/plain"))
                            .and_then(|p| p["body"]["data"].as_str())
                    })
            })
            .map(|data| {
                use base64::Engine;
                let cleaned = data.replace('-', "+").replace('_', "/");
                base64::engine::general_purpose::STANDARD
                    .decode(&cleaned)
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
            })
            .flatten();

        Ok(json!({
            "id": message_id,
            "subject": get_header("Subject"),
            "from": get_header("From"),
            "to": get_header("To"),
            "date": get_header("Date"),
            "body": body,
            "snippet": response["snippet"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Send Message Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for sending Gmail messages
pub struct GmailSendMessageTool {
    client: GmailClient,
}

impl GmailSendMessageTool {
    pub fn new(client: GmailClient) -> Self {
        Self { client }
    }

    pub fn from_env() -> crate::Result<Self> {
        Ok(Self::new(GmailClient::from_env()?))
    }
}

#[async_trait]
impl Tool for GmailSendMessageTool {
    fn name(&self) -> &str {
        "gmail_send_message"
    }

    fn description(&self) -> &str {
        "Send an email via Gmail."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                }
            },
            "required": ["to", "subject", "body"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let to = input["to"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'to' parameter".into()))?;
        let subject = input["subject"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'subject' parameter".into()))?;
        let body = input["body"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'body' parameter".into()))?;

        // Create RFC 2822 formatted message
        let raw_message = format!(
            "To: {}\r\nSubject: {}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{}",
            to, subject, body
        );

        // Base64url encode
        use base64::Engine;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(raw_message.as_bytes());

        let response = self.client.post(
            "/users/me/messages/send",
            json!({ "raw": encoded })
        ).await?;

        Ok(json!({
            "success": true,
            "message_id": response["id"],
            "thread_id": response["threadId"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gmail Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all Gmail tools with a registry
pub fn register_gmail_tools(registry: &mut ToolRegistry, access_token: impl Into<String>) {
    let client = GmailClient::new(access_token);
    registry.register(GmailListMessagesTool::new(client.clone()));
    registry.register(GmailReadMessageTool::new(client.clone()));
    registry.register(GmailSendMessageTool::new(client));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmail_client_creation() {
        let client = GmailClient::new("test-token");
        assert_eq!(client.access_token, "test-token");
    }
}
