//! DuckDuckGo search toolkit.
//!
//! Provides web search and news search via DuckDuckGo's HTML interface.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// DuckDuckGo search result
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub href: String,
    pub body: String,
}

/// Configuration for DuckDuckGo tools
#[derive(Clone)]
pub struct DuckDuckGoConfig {
    pub max_results: usize,
    pub timeout_secs: u64,
}

impl Default for DuckDuckGoConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            timeout_secs: 10,
        }
    }
}

/// Create a DuckDuckGo toolkit with search and news tools
pub fn duckduckgo_toolkit(config: DuckDuckGoConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(DuckDuckGoSearchTool {
        config: config.clone(),
    });
    registry.register(DuckDuckGoNewsTool { config });
    registry
}

struct DuckDuckGoSearchTool {
    config: DuckDuckGoConfig,
}

#[async_trait]
impl Tool for DuckDuckGoSearchTool {
    fn name(&self) -> &str {
        "duckduckgo_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo. Expects {\"query\": string, \"max_results\": number (optional)}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `query` for duckduckgo_search".into()))?;

        let max_results = input
            .get("max_results")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(self.config.max_results);

        let results = search_duckduckgo(query, max_results, self.config.timeout_secs).await?;
        Ok(json!({ "query": query, "results": results }))
    }
}

struct DuckDuckGoNewsTool {
    config: DuckDuckGoConfig,
}

#[async_trait]
impl Tool for DuckDuckGoNewsTool {
    fn name(&self) -> &str {
        "duckduckgo_news"
    }

    fn description(&self) -> &str {
        "Get latest news from DuckDuckGo. Expects {\"query\": string, \"max_results\": number (optional)}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `query` for duckduckgo_news".into()))?;

        let max_results = input
            .get("max_results")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(self.config.max_results);

        // For news, we append "news" to the query
        let results =
            search_duckduckgo(&format!("{} news", query), max_results, self.config.timeout_secs)
                .await?;
        Ok(json!({ "query": query, "results": results }))
    }
}

/// Perform a DuckDuckGo search using the HTML interface
async fn search_duckduckgo(
    query: &str,
    max_results: usize,
    timeout_secs: u64,
) -> Result<Vec<SearchResult>> {
    use std::time::Duration;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent("Mozilla/5.0 (compatible; AgnoBot/1.0)")
        .build()
        .map_err(|e| AgnoError::ToolInvocation {
            name: "duckduckgo_search".into(),
            source: Box::new(e),
        })?;

    // Use DuckDuckGo HTML endpoint
    let url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        urlencoding::encode(query)
    );

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| AgnoError::ToolInvocation {
            name: "duckduckgo_search".into(),
            source: Box::new(e),
        })?;

    let html = response
        .text()
        .await
        .map_err(|e| AgnoError::ToolInvocation {
            name: "duckduckgo_search".into(),
            source: Box::new(e),
        })?;

    // Parse results from HTML (simple extraction)
    let results = parse_duckduckgo_html(&html, max_results);
    Ok(results)
}

/// Parse DuckDuckGo HTML response to extract search results
fn parse_duckduckgo_html(html: &str, max_results: usize) -> Vec<SearchResult> {
    let mut results = Vec::new();

    // Simple regex-free parsing for DuckDuckGo HTML results
    // Results are in <a class="result__a" href="...">title</a>
    // with <a class="result__snippet">body</a>

    for (i, chunk) in html.split("result__a").enumerate() {
        if i == 0 || results.len() >= max_results {
            continue;
        }

        // Extract href
        let href = chunk
            .split("href=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap_or("")
            .to_string();

        // Extract title (text between > and </a>)
        let title = chunk
            .split('>')
            .nth(1)
            .and_then(|s| s.split("</a>").next())
            .unwrap_or("")
            .to_string();

        if !href.is_empty() && !title.is_empty() && href.starts_with("http") {
            results.push(SearchResult {
                title: html_decode(&title),
                href,
                body: String::new(), // Body requires more complex parsing
            });
        }
    }

    results
}

/// Simple HTML entity decoding
fn html_decode(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_duckduckgo_search_tool() {
        let config = DuckDuckGoConfig::default();
        let registry = duckduckgo_toolkit(config);
        assert!(registry.get("duckduckgo_search").is_some());
        assert!(registry.get("duckduckgo_news").is_some());
    }
}
