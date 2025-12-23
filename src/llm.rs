//! Language model implementations and abstractions.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::ModelConfig;
use crate::error::{AgnoError, Result};
use crate::message::{Message, Role, ToolCall};
use crate::tool::ToolDescription;

/// Result of a chat completion request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelCompletion {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Minimal abstraction around a chat completion provider.
#[async_trait]
pub trait LanguageModel: Send + Sync {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion>;
}

fn coalesce_error(status: reqwest::StatusCode, body: &str, provider: &str) -> AgnoError {
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return AgnoError::LanguageModel(format!("{provider} rate limit exceeded: {body}"));
    }
    AgnoError::LanguageModel(format!("{provider} request failed with {}: {body}", status))
}

fn serialize_tool_arguments(args: &Value) -> String {
    serde_json::to_string(args).unwrap_or_else(|_| args.to_string())
}

#[derive(Clone)]
pub struct OpenAIClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    base_url: String,
    organization: Option<String>,
}

impl OpenAIClient {
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        let api_key = cfg
            .openai
            .api_key
            .clone()
            .or_else(|| cfg.api_key.clone())
            .ok_or_else(|| {
                AgnoError::LanguageModel("missing OpenAI API key in model config".into())
            })?;
        let base_url = cfg
            .openai
            .endpoint
            .clone()
            .or_else(|| cfg.base_url.clone())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|err| AgnoError::LanguageModel(format!("http client error: {err}")))?,
            model: cfg.model.clone(),
            api_key,
            base_url,
            organization: cfg
                .openai
                .organization
                .clone()
                .or_else(|| cfg.organization.clone()),
        })
    }

    fn to_openai_messages(&self, messages: &[Message]) -> Vec<OpenAiMessage> {
        let mut built = Vec::new();
        for message in messages {
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            }
            .to_string();

            let mut tool_calls = None;
            if let Some(call) = &message.tool_call {
                tool_calls = Some(vec![OpenAiToolCall {
                    id: call.id.clone(),
                    r#type: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: call.name.clone(),
                        arguments: serialize_tool_arguments(&call.arguments),
                    },
                }]);
            }

            let content = if message.role == Role::Tool {
                message
                    .tool_result
                    .as_ref()
                    .map(|result| serialize_tool_arguments(&result.output))
                    .or_else(|| Some(message.content.clone()))
            } else {
                Some(message.content.clone())
            };

            let tool_call_id = message
                .tool_result
                .as_ref()
                .and_then(|result| result.tool_call_id.clone());

            built.push(OpenAiMessage {
                role,
                content,
                tool_call_id,
                tool_calls,
            });
        }
        built
    }

    fn to_openai_tools(&self, tools: &[ToolDescription]) -> Option<Vec<OpenAiTool>> {
        if tools.is_empty() {
            return None;
        }

        Some(
            tools
                .iter()
                .map(|tool| OpenAiTool {
                    r#type: "function".to_string(),
                    function: OpenAiFunction {
                        name: tool.name.clone(),
                        description: Some(tool.description.clone()),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect(),
        )
    }
}

#[async_trait]
impl LanguageModel for OpenAIClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion> {
        let payload = json!({
            "model": self.model,
            "messages": self.to_openai_messages(messages),
            "tools": self.to_openai_tools(tools),
            "tool_choice": if tools.is_empty() { Value::Null } else { Value::String("auto".to_string()) },
            "stream": stream,
        });

        let mut builder = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key),
            );
        if let Some(org) = &self.organization {
            builder = builder.header("OpenAI-Organization", org);
        }
        let resp = builder
            .json(&payload)
            .send()
            .await
            .map_err(|err| AgnoError::LanguageModel(format!("OpenAI request error: {err}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "openai"));
        }

        if stream {
            let mut content = String::new();
            let mut tool_calls: HashMap<String, OpenAiToolCallState> = HashMap::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|err| {
                    AgnoError::LanguageModel(format!("OpenAI stream error: {err}"))
                })?;
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" {
                        continue;
                    }
                    let parsed: OpenAiStreamChunk = serde_json::from_str(data).map_err(|err| {
                        AgnoError::LanguageModel(format!(
                            "OpenAI stream parse error `{data}`: {err}"
                        ))
                    })?;

                    for choice in parsed.choices {
                        if let Some(delta_content) = choice.delta.content {
                            content.push_str(&delta_content);
                        }
                        if let Some(calls) = choice.delta.tool_calls {
                            for delta_call in calls {
                                let id = delta_call
                                    .id
                                    .clone()
                                    .unwrap_or_else(|| format!("call_{}", tool_calls.len()));
                                let state = tool_calls.entry(id.clone()).or_default();
                                if let Some(function) = delta_call.function {
                                    if let Some(name) = function.name {
                                        state.name = Some(name);
                                    }
                                    if let Some(args) = function.arguments {
                                        state.arguments.push_str(&args);
                                    }
                                }
                                state.id = Some(id);
                            }
                        }
                    }
                }
            }

            let calls: Vec<ToolCall> = tool_calls
                .into_values()
                .filter_map(|state| {
                    let name = state.name?;
                    let args = serde_json::from_str(&state.arguments)
                        .unwrap_or_else(|_| Value::String(state.arguments.clone()));
                    Some(ToolCall {
                        id: state.id,
                        name,
                        arguments: args,
                    })
                })
                .collect();

            return Ok(ModelCompletion {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: calls,
            });
        }

        let body: OpenAiResponse = resp.json().await.map_err(|err| {
            AgnoError::LanguageModel(format!("OpenAI response parse error: {err}"))
        })?;

        let first = body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AgnoError::LanguageModel("OpenAI returned no choices".into()))?;

        let mut tool_calls = Vec::new();
        if let Some(calls) = first.message.tool_calls {
            for call in calls {
                let args = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                tool_calls.push(ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: args,
                });
            }
        }

        Ok(ModelCompletion {
            content: first.message.content,
            tool_calls,
        })
    }
}

#[derive(Clone)]
pub struct AnthropicClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    endpoint: String,
}

impl AnthropicClient {
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        let api_key = cfg
            .anthropic
            .api_key
            .clone()
            .or_else(|| cfg.api_key.clone())
            .ok_or_else(|| {
                AgnoError::LanguageModel("missing Anthropic API key in model config".into())
            })?;
        let endpoint = cfg
            .anthropic
            .endpoint
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com/v1/messages".to_string());
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|err| AgnoError::LanguageModel(format!("http client error: {err}")))?,
            model: cfg.model.clone(),
            api_key,
            endpoint,
        })
    }

    fn to_messages(&self, messages: &[Message]) -> Vec<AnthropicMessage> {
        messages
            .iter()
            .filter_map(|message| match message.role {
                Role::System => None,
                Role::User | Role::Assistant | Role::Tool => Some(AnthropicMessage {
                    role: match message.role {
                        Role::User => "user",
                        Role::Assistant | Role::Tool => "assistant",
                        Role::System => unreachable!(),
                    }
                    .to_string(),
                    content: vec![AnthropicContentBlock {
                        r#type: "text".to_string(),
                        text: Some(message.content.clone()),
                        name: None,
                        input_schema: None,
                    }],
                }),
            })
            .collect()
    }

    fn to_tools(&self, tools: &[ToolDescription]) -> Option<Vec<AnthropicTool>> {
        if tools.is_empty() {
            return None;
        }
        Some(
            tools
                .iter()
                .map(|tool| AnthropicTool {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    input_schema: tool
                        .parameters
                        .clone()
                        .unwrap_or_else(|| json!({"type":"object"})),
                })
                .collect(),
        )
    }
}

#[async_trait]
impl LanguageModel for AnthropicClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion> {
        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());
        let payload = json!({
            "model": self.model,
            "system": system,
            "messages": self.to_messages(messages),
            "tools": self.to_tools(tools),
            "stream": stream,
        });

        let resp = self
            .http
            .post(&self.endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&payload)
            .send()
            .await
            .map_err(|err| AgnoError::LanguageModel(format!("Anthropic request error: {err}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "anthropic"));
        }

        if stream {
            let mut content = String::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|err| {
                    AgnoError::LanguageModel(format!("Anthropic stream error: {err}"))
                })?;
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" || data.is_empty() {
                        continue;
                    }
                    let parsed: AnthropicStreamChunk =
                        serde_json::from_str(data).map_err(|err| {
                            AgnoError::LanguageModel(format!(
                                "Anthropic stream parse error `{data}`: {err}"
                            ))
                        })?;
                    if let Some(text) = parsed.delta.text {
                        content.push_str(&text);
                    }
                }
            }

            return Ok(ModelCompletion {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: Vec::new(),
            });
        }

        let parsed: AnthropicResponse = resp.json().await.map_err(|err| {
            AgnoError::LanguageModel(format!("Anthropic response parse error: {err}"))
        })?;

        let content = parsed
            .content
            .iter()
            .filter_map(|block| block.text.clone())
            .collect::<Vec<String>>()
            .join("");

        Ok(ModelCompletion {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    endpoint: String,
}

impl GeminiClient {
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        let api_key = cfg
            .gemini
            .api_key
            .clone()
            .or_else(|| cfg.api_key.clone())
            .ok_or_else(|| {
                AgnoError::LanguageModel("missing Gemini API key in model config".into())
            })?;
        let endpoint = cfg
            .gemini
            .endpoint
            .clone()
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|err| AgnoError::LanguageModel(format!("http client error: {err}")))?,
            model: cfg.model.clone(),
            api_key,
            endpoint,
        })
    }

    fn to_contents(&self, messages: &[Message]) -> Vec<GeminiMessage> {
        messages
            .iter()
            .filter_map(|message| {
                let role = match message.role {
                    Role::User => "user",
                    Role::Assistant => "model",
                    Role::System => "system",
                    Role::Tool => "user",
                };
                Some(GeminiMessage {
                    role: role.to_string(),
                    parts: vec![GeminiPart {
                        text: message.content.clone(),
                    }],
                })
            })
            .collect()
    }
}

#[async_trait]
impl LanguageModel for GeminiClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDescription],
        _stream: bool,
    ) -> Result<ModelCompletion> {
        let payload = json!({
            "contents": self.to_contents(messages),
        });
        let resp = self
            .http
            .post(format!(
                "{}/models/{}:generateContent?key={}",
                self.endpoint, self.model, self.api_key
            ))
            .json(&payload)
            .send()
            .await
            .map_err(|err| AgnoError::LanguageModel(format!("Gemini request error: {err}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "gemini"));
        }

        let parsed: GeminiResponse = resp.json().await.map_err(|err| {
            AgnoError::LanguageModel(format!("Gemini response parse error: {err}"))
        })?;

        let content = parsed
            .candidates
            .get(0)
            .and_then(|cand| cand.content.parts.get(0))
            .map(|part| part.text.clone())
            .unwrap_or_default();

        Ok(ModelCompletion {
            content: if content.is_empty() {
                None
            } else {
                Some(content)
            },
            tool_calls: Vec::new(),
        })
    }
}

#[derive(Clone)]
pub struct CohereClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    endpoint: String,
}

impl CohereClient {
    pub fn from_config(cfg: &ModelConfig) -> Result<Self> {
        let api_key = cfg
            .cohere
            .api_key
            .clone()
            .or_else(|| cfg.api_key.clone())
            .ok_or_else(|| {
                AgnoError::LanguageModel("missing Cohere API key in model config".into())
            })?;
        let endpoint = cfg
            .cohere
            .endpoint
            .clone()
            .unwrap_or_else(|| "https://api.cohere.ai/v2/chat".to_string());
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(|err| AgnoError::LanguageModel(format!("http client error: {err}")))?,
            model: cfg.model.clone(),
            api_key,
            endpoint,
        })
    }

    fn to_messages(&self, messages: &[Message]) -> Vec<CohereMessage> {
        messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                CohereMessage {
                    role: role.to_string(),
                    content: message.content.clone(),
                }
            })
            .collect()
    }

    fn to_tools(&self, tools: &[ToolDescription]) -> Option<Vec<CohereTool>> {
        if tools.is_empty() {
            return None;
        }
        Some(
            tools
                .iter()
                .map(|tool| CohereTool {
                    r#type: "function".to_string(),
                    function: CohereFunction {
                        name: tool.name.clone(),
                        description: Some(tool.description.clone()),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect(),
        )
    }
}

#[async_trait]
impl LanguageModel for CohereClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion> {
        let payload = json!({
            "model": self.model,
            "messages": self.to_messages(messages),
            "tools": self.to_tools(tools),
            "stream": stream,
        });

        let resp = self
            .http
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|err| AgnoError::LanguageModel(format!("Cohere request error: {err}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "cohere"));
        }

        if stream {
            let mut content = String::new();
            let tool_calls_map: HashMap<String, OpenAiToolCallState> = HashMap::new();
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|err| {
                    AgnoError::LanguageModel(format!("Cohere stream error: {err}"))
                })?;
                let text = String::from_utf8_lossy(&chunk);
                for line in text.lines() {
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" || data.is_empty() {
                        continue;
                    }
                    if let Ok(parsed) = serde_json::from_str::<CohereStreamChunk>(data) {
                        if let Some(delta) = parsed.delta {
                            if let Some(msg) = delta.message {
                                if let Some(c) = msg.content {
                                    if let Some(text_content) = c.get("text") {
                                        if let Some(t) = text_content.as_str() {
                                            content.push_str(t);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let calls: Vec<ToolCall> = tool_calls_map
                .into_values()
                .filter_map(|state| {
                    let name = state.name?;
                    let args = serde_json::from_str(&state.arguments)
                        .unwrap_or_else(|_| Value::String(state.arguments.clone()));
                    Some(ToolCall {
                        id: state.id,
                        name,
                        arguments: args,
                    })
                })
                .collect();

            return Ok(ModelCompletion {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls: calls,
            });
        }

        let body: CohereResponse = resp.json().await.map_err(|err| {
            AgnoError::LanguageModel(format!("Cohere response parse error: {err}"))
        })?;

        let content = body.message.and_then(|m| {
            m.content.and_then(|c| c.get("text").and_then(|v| v.as_str().map(|s| s.to_string())))
        });

        let mut tool_calls = Vec::new();
        if let Some(calls) = body.tool_calls {
            for call in calls {
                let args = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                tool_calls.push(ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments: args,
                });
            }
        }

        Ok(ModelCompletion {
            content,
            tool_calls,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Groq Client (OpenAI-compatible API)
// ─────────────────────────────────────────────────────────────────────────────

/// Groq client - uses OpenAI-compatible API with Groq's endpoint.
/// Default model: llama-3.3-70b-versatile
#[derive(Clone)]
pub struct GroqClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    base_url: String,
}

impl GroqClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build http client"),
            model: "llama-3.3-70b-versatile".to_string(),
            api_key: api_key.into(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| AgnoError::LanguageModel("GROQ_API_KEY not set".into()))?;
        Ok(Self::new(api_key))
    }
}

#[async_trait]
impl LanguageModel for GroqClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion> {
        // Convert messages to OpenAI format
        let oai_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let mut msg = json!({
                    "role": role,
                    "content": m.content.clone()
                });
                if let Some(ref result) = m.tool_result {
                    if let Some(ref call_id) = result.tool_call_id {
                        msg["tool_call_id"] = json!(call_id);
                    }
                }
                msg
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": oai_messages,
            "stream": stream
        });

        if !tools.is_empty() {
            let oai_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(oai_tools);
        }

        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Groq request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "Groq"));
        }

        let json: Value = resp
            .json()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Groq parse error: {e}")))?;

        let choice = &json["choices"][0]["message"];
        let content = choice["content"].as_str().map(String::from);

        let mut tool_calls = Vec::new();
        if let Some(calls) = choice["tool_calls"].as_array() {
            for call in calls {
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let args_str = call["function"]["arguments"].as_str().unwrap_or("{}");
                let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                tool_calls.push(ToolCall {
                    id: call["id"].as_str().map(String::from),
                    name,
                    arguments: args,
                });
            }
        }

        Ok(ModelCompletion { content, tool_calls })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama Client (Local LLM)
// ─────────────────────────────────────────────────────────────────────────────

/// Ollama client for local LLM inference.
/// Default model: llama3.1
#[derive(Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    model: String,
    base_url: String,
}

impl OllamaClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(300)) // Local models can be slow
                .build()
                .expect("failed to build http client"),
            model: "llama3.1".to_string(),
            base_url: "http://localhost:11434".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.base_url = host.into();
        self
    }

    pub fn from_env() -> Self {
        let mut client = Self::new();
        if let Ok(host) = std::env::var("OLLAMA_HOST") {
            client.base_url = host;
        }
        if let Ok(model) = std::env::var("OLLAMA_MODEL") {
            client.model = model;
        }
        client
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LanguageModel for OllamaClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        _stream: bool,
    ) -> Result<ModelCompletion> {
        // Convert messages to Ollama format
        let ollama_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                json!({
                    "role": role,
                    "content": m.content.clone()
                })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": ollama_messages,
            "stream": false
        });

        if !tools.is_empty() {
            let ollama_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = json!(ollama_tools);
        }

        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Ollama request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "Ollama"));
        }

        let json: Value = resp
            .json()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Ollama parse error: {e}")))?;

        let message = &json["message"];
        let content = message["content"].as_str().map(String::from);

        let mut tool_calls = Vec::new();
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                let func = &call["function"];
                let name = func["name"].as_str().unwrap_or("").to_string();
                let args = func["arguments"].clone();
                tool_calls.push(ToolCall {
                    id: None,
                    name,
                    arguments: args,
                });
            }
        }

        Ok(ModelCompletion { content, tool_calls })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mistral AI Client
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral AI client using their OpenAI-compatible API.
/// Default model: mistral-large-latest
#[derive(Clone)]
pub struct MistralClient {
    http: reqwest::Client,
    model: String,
    api_key: String,
    base_url: String,
}

impl MistralClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build http client"),
            model: "mistral-large-latest".to_string(),
            api_key: api_key.into(),
            base_url: "https://api.mistral.ai/v1".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| AgnoError::LanguageModel("MISTRAL_API_KEY not set".into()))?;
        Ok(Self::new(api_key))
    }
}

#[async_trait]
impl LanguageModel for MistralClient {
    async fn complete_chat(
        &self,
        messages: &[Message],
        tools: &[ToolDescription],
        stream: bool,
    ) -> Result<ModelCompletion> {
        // Convert messages to Mistral format (OpenAI-compatible)
        let mistral_messages: Vec<Value> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };

                let mut msg = json!({
                    "role": role,
                    "content": m.content.clone()
                });

                // Add tool_call_id for tool responses
                if m.role == Role::Tool {
                    if let Some(ref tc) = m.tool_call {
                        if let Some(ref id) = tc.id {
                            msg["tool_call_id"] = json!(id);
                        }
                    }
                }

                // Add tool_calls for assistant messages
                if let Some(ref tc) = m.tool_call {
                    if m.role == Role::Assistant {
                        msg["tool_calls"] = json!([{
                            "id": tc.id.clone().unwrap_or_default(),
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": serialize_tool_arguments(&tc.arguments)
                            }
                        }]);
                        msg["content"] = json!(null);
                    }
                }

                msg
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "messages": mistral_messages,
            "stream": stream
        });

        if !tools.is_empty() {
            let mistral_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters.clone().unwrap_or(json!({"type": "object", "properties": {}}))
                        }
                    })
                })
                .collect();
            body["tools"] = json!(mistral_tools);
            body["tool_choice"] = json!("auto");
        }

        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Mistral request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(coalesce_error(status, &body, "Mistral"));
        }

        // Parse response (OpenAI-compatible format)
        let json: Value = resp
            .json()
            .await
            .map_err(|e| AgnoError::LanguageModel(format!("Mistral parse error: {e}")))?;

        let choice = json["choices"]
            .as_array()
            .and_then(|c| c.first())
            .ok_or_else(|| AgnoError::LanguageModel("Mistral returned no choices".into()))?;

        let message = &choice["message"];
        let content = message["content"].as_str().map(String::from);

        let mut tool_calls = Vec::new();
        if let Some(calls) = message["tool_calls"].as_array() {
            for call in calls {
                let id = call["id"].as_str().map(String::from);
                let func = &call["function"];
                let name = func["name"].as_str().unwrap_or("").to_string();
                let args_str = func["arguments"].as_str().unwrap_or("{}");
                let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: args,
                });
            }
        }

        Ok(ModelCompletion { content, tool_calls })
    }
}

/// A deterministic model used for tests and demos.

pub struct StubModel {
    responses: Mutex<VecDeque<String>>,
}


impl StubModel {
    pub fn new(responses: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum StubDirective {
    Respond { content: String },
    CallTool { name: String, arguments: Value },
}

#[async_trait]
impl LanguageModel for StubModel {
    async fn complete_chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDescription],
        _stream: bool,
    ) -> Result<ModelCompletion> {
        let mut locked = self.responses.lock().expect("stub model poisoned");
        let raw = locked.pop_front().ok_or_else(|| {
            AgnoError::LanguageModel("StubModel ran out of scripted responses".into())
        })?;

        match serde_json::from_str::<StubDirective>(&raw) {
            Ok(StubDirective::Respond { content }) => Ok(ModelCompletion {
                content: Some(content),
                tool_calls: Vec::new(),
            }),
            Ok(StubDirective::CallTool { name, arguments }) => Ok(ModelCompletion {
                content: None,
                tool_calls: vec![ToolCall {
                    id: None,
                    name,
                    arguments,
                }],
            }),
            Err(_) => Ok(ModelCompletion {
                content: Some(raw),
                tool_calls: Vec::new(),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    r#type: String,
    function: OpenAiFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiTool {
    r#type: String,
    function: OpenAiFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAiFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Default)]
struct OpenAiToolCallState {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiDeltaChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDeltaChoice {
    delta: OpenAiDelta,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCallDelta {
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicContentBlock {
    r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_schema: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamChunk {
    delta: AnthropicDelta,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiMessage {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiCandidateContent,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidateContent {
    parts: Vec<GeminiPart>,
}

// Cohere data structures
#[derive(Debug, Serialize, Deserialize)]
struct CohereMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CohereTool {
    r#type: String,
    function: CohereFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct CohereFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CohereResponse {
    #[serde(default)]
    message: Option<CohereResponseMessage>,
    #[serde(default)]
    tool_calls: Option<Vec<CohereToolCall>>,
}

#[derive(Debug, Deserialize)]
struct CohereResponseMessage {
    #[serde(default)]
    content: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CohereToolCall {
    #[serde(default)]
    id: Option<String>,
    function: CohereFunctionCall,
}

#[derive(Debug, Deserialize)]
struct CohereFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct CohereStreamChunk {
    #[serde(default)]
    delta: Option<CohereDelta>,
}

#[derive(Debug, Deserialize)]
struct CohereDelta {
    #[serde(default)]
    message: Option<CohereResponseMessage>,
}

