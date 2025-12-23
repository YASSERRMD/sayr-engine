//! HTTP API toolkit.
//!
//! Provides tools for making HTTP requests to external APIs.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// Configuration for HTTP API tools
#[derive(Clone)]
pub struct HttpApiConfig {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub default_headers: HashMap<String, String>,
    pub timeout_secs: u64,
    pub verify_ssl: bool,
}

impl Default for HttpApiConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key: None,
            default_headers: HashMap::new(),
            timeout_secs: 30,
            verify_ssl: true,
        }
    }
}

impl HttpApiConfig {
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.insert(key.into(), value.into());
        self
    }
}

/// Create an HTTP API toolkit
pub fn http_api_toolkit(config: HttpApiConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(HttpRequestTool { config });
    registry
}

struct HttpRequestTool {
    config: HttpApiConfig,
}

#[derive(Debug, Deserialize)]
struct HttpRequestInput {
    endpoint: String,
    #[serde(default = "default_method")]
    method: String,
    #[serde(default)]
    params: Option<HashMap<String, String>>,
    #[serde(default)]
    headers: Option<HashMap<String, String>>,
    #[serde(default)]
    body: Option<Value>,
}

fn default_method() -> String {
    "GET".to_string()
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Make an HTTP request. Expects {\"endpoint\": string, \"method\": \"GET\"|\"POST\"|\"PUT\"|\"DELETE\"|\"PATCH\", \"params\": object, \"headers\": object, \"body\": object}."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "URL or path to request"},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                "params": {"type": "object", "description": "Query parameters"},
                "headers": {"type": "object", "description": "Additional headers"},
                "body": {"type": "object", "description": "JSON body for POST/PUT/PATCH"}
            },
            "required": ["endpoint"]
        }))
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let req: HttpRequestInput = serde_json::from_value(input).map_err(|e| {
            AgnoError::Protocol(format!("Invalid http_request input: {}", e))
        })?;

        // Build URL
        let url = if let Some(ref base) = self.config.base_url {
            format!(
                "{}/{}",
                base.trim_end_matches('/'),
                req.endpoint.trim_start_matches('/')
            )
        } else {
            req.endpoint.clone()
        };

        // Build client
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.config.timeout_secs))
            .danger_accept_invalid_certs(!self.config.verify_ssl)
            .build()
            .map_err(|e| AgnoError::ToolInvocation {
                name: "http_request".into(),
                source: Box::new(e),
            })?;

        // Build request
        let method = req.method.to_uppercase();
        let mut request = match method.as_str() {
            "GET" => client.get(&url),
            "POST" => client.post(&url),
            "PUT" => client.put(&url),
            "DELETE" => client.delete(&url),
            "PATCH" => client.patch(&url),
            _ => {
                return Ok(json!({
                    "error": format!("Unsupported HTTP method: {}", method)
                }))
            }
        };

        // Add query params
        if let Some(params) = req.params {
            request = request.query(&params);
        }

        // Build headers
        let mut headers = HeaderMap::new();
        for (k, v) in &self.config.default_headers {
            if let (Ok(name), Ok(value)) = (
                HeaderName::try_from(k.as_str()),
                HeaderValue::from_str(v),
            ) {
                headers.insert(name, value);
            }
        }

        // Add API key if configured
        if let Some(ref api_key) = self.config.api_key {
            if let Ok(value) = HeaderValue::from_str(&format!("Bearer {}", api_key)) {
                headers.insert(reqwest::header::AUTHORIZATION, value);
            }
        }

        // Add custom headers
        if let Some(custom_headers) = req.headers {
            for (k, v) in custom_headers {
                if let (Ok(name), Ok(value)) = (
                    HeaderName::try_from(k.as_str()),
                    HeaderValue::from_str(&v),
                ) {
                    headers.insert(name, value);
                }
            }
        }

        request = request.headers(headers);

        // Add body for POST/PUT/PATCH
        if let Some(body) = req.body {
            if ["POST", "PUT", "PATCH"].contains(&method.as_str()) {
                request = request.json(&body);
            }
        }

        // Execute request
        let response = request.send().await.map_err(|e| AgnoError::ToolInvocation {
            name: "http_request".into(),
            source: Box::new(e),
        })?;

        let status = response.status().as_u16();
        let response_headers: HashMap<String, String> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        // Parse response body
        let body_text = response.text().await.unwrap_or_default();
        let body_json: Value = serde_json::from_str(&body_text)
            .unwrap_or_else(|_| json!({ "text": body_text }));

        Ok(json!({
            "status_code": status,
            "headers": response_headers,
            "data": body_json,
            "success": status >= 200 && status < 300
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_http_api_toolkit_creation() {
        let config = HttpApiConfig::default()
            .with_base_url("https://api.example.com")
            .with_api_key("test-key");
        let registry = http_api_toolkit(config);
        assert!(registry.get("http_request").is_some());
    }
}
