//! MCP (Model Context Protocol) client support for agno-rust.
//!
//! This module provides integration with MCP servers, allowing agents to
//! access tools, resources, and prompts exposed by MCP-compatible servers.
//!
//! MCP is a protocol for connecting AI models to external tools and data sources.
//!
//! # Transport Types
//! - **Stdio**: Launch an MCP server as a subprocess and communicate via stdin/stdout
//! - **HTTP**: Connect to an MCP server via HTTP/SSE
//!
//! # Example
//! ```rust,ignore
//! use agno_rust::mcp::{McpClient, StdioTransport};
//!
//! let transport = StdioTransport::new("npx", &["-y", "@modelcontextprotocol/server-filesystem", "."]);
//! let client = McpClient::new(transport);
//! let tools = client.list_tools().await?;
//! ```

use crate::error::AgnoError;
use crate::tool::{Tool, ToolRegistry};
use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// MCP Protocol Types
// ─────────────────────────────────────────────────────────────────────────────

/// JSON-RPC request structure
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC response structure
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

/// MCP Tool definition from a server
#[derive(Debug, Clone, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP list_tools response
#[derive(Debug, Clone, Deserialize)]
pub struct ListToolsResult {
    pub tools: Vec<McpToolDefinition>,
}

/// MCP call_tool result content
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType", default)]
        mime_type: Option<String>,
    },
    #[serde(rename = "resource")]
    Resource { resource: Value },
}

/// MCP call_tool result
#[derive(Debug, Clone, Deserialize)]
pub struct CallToolResult {
    pub content: Vec<ContentItem>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// MCP server capabilities
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerCapabilities {
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub resources: Option<Value>,
    #[serde(default)]
    pub prompts: Option<Value>,
}

/// MCP initialize result
#[derive(Debug, Clone, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// MCP server info
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Transport Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Transport layer for MCP communication
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and receive a response
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse>;

    /// Close the transport
    async fn close(&self) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP Transport
// ─────────────────────────────────────────────────────────────────────────────

/// Transport that communicates with an MCP server via HTTP
pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
    request_id: AtomicU64,
}

impl HttpTransport {
    /// Create a new HTTP transport
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.into(),
            request_id: AtomicU64::new(1),
        }
    }

    /// Create with custom headers (e.g., for authentication)
    pub fn with_headers(url: impl Into<String>, headers: HashMap<String, String>) -> Self {
        let mut header_map = reqwest::header::HeaderMap::new();
        for (key, value) in headers {
            if let (Ok(name), Ok(val)) = (
                reqwest::header::HeaderName::try_from(key),
                reqwest::header::HeaderValue::try_from(value),
            ) {
                header_map.insert(name, val);
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(header_map)
            .build()
            .unwrap_or_default();

        Self {
            client,
            url: url.into(),
            request_id: AtomicU64::new(1),
        }
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn send(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        // Assign a unique request ID
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        request.id = id;

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgnoError::Mcp(format!("HTTP request failed: {}", e)))?;

        let response_json: JsonRpcResponse = response
            .json()
            .await
            .map_err(|e| AgnoError::Mcp(format!("Failed to parse response: {}", e)))?;

        Ok(response_json)
    }

    async fn close(&self) -> Result<()> {
        // HTTP transport doesn't need explicit cleanup
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stdio Transport (Async via tokio::process)
// ─────────────────────────────────────────────────────────────────────────────

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

/// Transport that communicates with an MCP server via stdio
pub struct StdioTransport {
    #[allow(dead_code)]
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    stdout: Arc<Mutex<Option<BufReader<ChildStdout>>>>,
    request_id: AtomicU64,
}

impl StdioTransport {
    /// Create a new stdio transport by launching a subprocess
    pub fn new(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| AgnoError::Mcp(format!("Failed to spawn MCP server: {}", e)))?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take().map(BufReader::new);

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(stdout)),
            request_id: AtomicU64::new(1),
        })
    }

    /// Create from an existing command with environment variables
    pub fn with_env(command: &str, args: &[&str], env: HashMap<String, String>) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());

        for (key, value) in env {
            cmd.env(key, value);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| AgnoError::Mcp(format!("Failed to spawn MCP server: {}", e)))?;

        let stdin = child.stdin.take();
        let stdout = child.stdout.take().map(BufReader::new);

        Ok(Self {
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(stdout)),
            request_id: AtomicU64::new(1),
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, mut request: JsonRpcRequest) -> Result<JsonRpcResponse> {
        // Assign a unique request ID
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        request.id = id;

        // Serialize the request
        let request_json = serde_json::to_string(&request)
            .map_err(|e| AgnoError::Mcp(format!("Failed to serialize request: {}", e)))?;

        // Send request
        {
            let mut stdin_guard = self.stdin.lock().await;

            if let Some(ref mut stdin) = *stdin_guard {
                stdin
                    .write_all(request_json.as_bytes())
                    .await
                    .map_err(|e| AgnoError::Mcp(format!("Failed to write to MCP server: {}", e)))?;
                stdin
                    .write_all(b"\n")
                    .await
                    .map_err(|e| AgnoError::Mcp(format!("Failed to write newline: {}", e)))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| AgnoError::Mcp(format!("Failed to flush: {}", e)))?;
            } else {
                return Err(AgnoError::Mcp("MCP server stdin not available".into()));
            }
        }

        // Read response
        {
            let mut stdout_guard = self.stdout.lock().await;

            if let Some(ref mut stdout) = *stdout_guard {
                let mut line = String::new();
                stdout
                    .read_line(&mut line)
                    .await
                    .map_err(|e| AgnoError::Mcp(format!("Failed to read from MCP server: {}", e)))?;

                let response: JsonRpcResponse = serde_json::from_str(&line)
                    .map_err(|e| AgnoError::Mcp(format!("Failed to parse response: {}", e)))?;

                Ok(response)
            } else {
                Err(AgnoError::Mcp("MCP server stdout not available".into()))
            }
        }
    }

    async fn close(&self) -> Result<()> {
        let mut child_guard = self.child.lock().await;

        if let Some(ref mut child) = *child_guard {
            child
                .kill()
                .await
                .map_err(|e| AgnoError::Mcp(format!("Failed to kill MCP server: {}", e)))?;
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MCP Client
// ─────────────────────────────────────────────────────────────────────────────

/// MCP client for connecting to MCP servers
pub struct McpClient<T: McpTransport> {
    transport: T,
    initialized: bool,
    server_info: Option<ServerInfo>,
    #[allow(dead_code)]
    capabilities: Option<ServerCapabilities>,
}

impl<T: McpTransport> McpClient<T> {
    /// Create a new MCP client with the given transport
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            initialized: false,
            server_info: None,
            capabilities: None,
        }
    }

    /// Initialize the connection to the MCP server
    pub async fn initialize(&mut self) -> Result<&ServerInfo> {
        if self.initialized {
            return self
                .server_info
                .as_ref()
                .ok_or_else(|| AgnoError::Mcp("Server info not available".into()));
        }

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 0,
            method: "initialize".to_string(),
            params: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agno-rust",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
        };

        let response = self.transport.send(request).await?;

        if let Some(error) = response.error {
            return Err(AgnoError::Mcp(format!(
                "Initialize failed: {}",
                error.message
            )));
        }

        let result: InitializeResult = serde_json::from_value(response.result.unwrap_or_default())
            .map_err(|e| AgnoError::Mcp(format!("Failed to parse initialize result: {}", e)))?;

        self.server_info = Some(result.server_info);
        self.capabilities = Some(result.capabilities);
        self.initialized = true;

        // Send initialized notification
        let notification = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 0,
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let _ = self.transport.send(notification).await;

        self.server_info
            .as_ref()
            .ok_or_else(|| AgnoError::Mcp("Server info not available".into()))
    }

    /// List available tools from the MCP server
    pub async fn list_tools(&mut self) -> Result<Vec<McpToolDefinition>> {
        if !self.initialized {
            self.initialize().await?;
        }

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 0,
            method: "tools/list".to_string(),
            params: None,
        };

        let response = self.transport.send(request).await?;

        if let Some(error) = response.error {
            return Err(AgnoError::Mcp(format!(
                "list_tools failed: {}",
                error.message
            )));
        }

        let result: ListToolsResult = serde_json::from_value(response.result.unwrap_or_default())
            .map_err(|e| AgnoError::Mcp(format!("Failed to parse list_tools result: {}", e)))?;

        Ok(result.tools)
    }

    /// Call a tool on the MCP server
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<CallToolResult> {
        if !self.initialized {
            self.initialize().await?;
        }

        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 0,
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": name,
                "arguments": arguments
            })),
        };

        let response = self.transport.send(request).await?;

        if let Some(error) = response.error {
            return Err(AgnoError::Mcp(format!(
                "call_tool failed: {}",
                error.message
            )));
        }

        let result: CallToolResult = serde_json::from_value(response.result.unwrap_or_default())
            .map_err(|e| AgnoError::Mcp(format!("Failed to parse call_tool result: {}", e)))?;

        Ok(result)
    }

    /// Close the MCP client
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }

    /// Check if the client is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get server info if available
    pub fn server_info(&self) -> Option<&ServerInfo> {
        self.server_info.as_ref()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MCP Tools Integration
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper that creates agno tools from MCP server tools
pub struct McpTools<T: McpTransport + 'static> {
    client: Arc<tokio::sync::Mutex<McpClient<T>>>,
    tool_prefix: Option<String>,
}

impl<T: McpTransport + 'static> McpTools<T> {
    /// Create a new MCP tools wrapper
    pub fn new(client: McpClient<T>) -> Self {
        Self {
            client: Arc::new(tokio::sync::Mutex::new(client)),
            tool_prefix: None,
        }
    }

    /// Set a prefix for all tool names
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    /// Get tools and register them with a ToolRegistry
    pub async fn register_tools(&self, registry: &mut ToolRegistry) -> Result<usize> {
        let mut client = self.client.lock().await;
        let tools = client.list_tools().await?;

        let mut count = 0;
        for tool_def in tools {
            let name = if let Some(ref prefix) = self.tool_prefix {
                format!("{}_{}", prefix, tool_def.name)
            } else {
                tool_def.name.clone()
            };

            let description = tool_def
                .description
                .unwrap_or_else(|| format!("MCP tool: {}", tool_def.name));

            // Create a wrapper tool that calls the MCP server
            let client_clone = Arc::clone(&self.client);
            let tool_name = tool_def.name.clone();

            let wrapper = McpToolWrapper {
                name: name.clone(),
                description,
                parameters: tool_def.input_schema,
                client: client_clone,
                mcp_tool_name: tool_name,
            };

            registry.register(wrapper);
            count += 1;
        }

        Ok(count)
    }
}

/// Wrapper that implements Tool for an MCP tool
struct McpToolWrapper<T: McpTransport + 'static> {
    name: String,
    description: String,
    parameters: Value,
    client: Arc<tokio::sync::Mutex<McpClient<T>>>,
    mcp_tool_name: String,
}

#[async_trait]
impl<T: McpTransport + 'static> Tool for McpToolWrapper<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Option<serde_json::Value> {
        Some(self.parameters.clone())
    }

    async fn call(&self, input: serde_json::Value) -> crate::Result<serde_json::Value> {
        let mut client = self.client.lock().await;
        let result = client.call_tool(&self.mcp_tool_name, input).await?;

        // Convert result to JSON value
        if result.is_error {
            // Return error as a structured response
            return Ok(serde_json::json!({
                "error": true,
                "content": result.content.iter().map(|c| match c {
                    ContentItem::Text { text } => text.clone(),
                    ContentItem::Image { .. } => "[image]".to_string(),
                    ContentItem::Resource { .. } => "[resource]".to_string(),
                }).collect::<Vec<_>>().join("\n")
            }));
        }

        // Build successful response
        let mut text_content = Vec::new();
        let mut images = Vec::new();

        for item in result.content {
            match item {
                ContentItem::Text { text } => text_content.push(text),
                ContentItem::Image { data, mime_type } => {
                    images.push(serde_json::json!({
                        "type": "image",
                        "data": data,
                        "mimeType": mime_type.unwrap_or_else(|| "image/png".to_string())
                    }));
                }
                ContentItem::Resource { resource } => {
                    text_content.push(format!("[Resource: {}]", resource));
                }
            }
        }

        if images.is_empty() {
            Ok(serde_json::json!({
                "content": text_content.join("\n")
            }))
        } else {
            Ok(serde_json::json!({
                "content": text_content.join("\n"),
                "images": images
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_serialization() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: "tools/list".to_string(),
            params: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
    }

    #[test]
    fn test_json_rpc_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let response: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, 1);
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn test_mcp_tool_definition_deserialization() {
        let json = r#"{
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }"#;

        let tool: McpToolDefinition = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description, Some("Read a file from disk".to_string()));
    }

    #[test]
    fn test_http_transport_creation() {
        let transport = HttpTransport::new("http://localhost:3000/mcp");
        assert_eq!(transport.url, "http://localhost:3000/mcp");
    }
}
