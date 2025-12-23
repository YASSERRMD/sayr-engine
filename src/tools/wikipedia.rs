//! Wikipedia toolkit.
//!
//! Provides tools for searching Wikipedia and retrieving article summaries.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// Create a Wikipedia toolkit
pub fn wikipedia_toolkit() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(WikipediaSearchTool);
    registry
}

struct WikipediaSearchTool;

#[async_trait]
impl Tool for WikipediaSearchTool {
    fn name(&self) -> &str {
        "wikipedia_search"
    }

    fn description(&self) -> &str {
        "Search Wikipedia for a topic and get a summary. Expects {\"query\": string}."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to search on Wikipedia"}
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| AgnoError::Protocol("missing `query` for wikipedia_search".into()))?;

        // Use Wikipedia API to get summary
        let summary = fetch_wikipedia_summary(query).await?;

        Ok(json!({
            "query": query,
            "title": summary.title,
            "extract": summary.extract,
            "url": format!("https://en.wikipedia.org/wiki/{}", urlencoding::encode(&summary.title))
        }))
    }
}

#[derive(Debug)]
struct WikipediaSummary {
    title: String,
    extract: String,
}

async fn fetch_wikipedia_summary(query: &str) -> Result<WikipediaSummary> {
    let client = reqwest::Client::new();

    // Use Wikipedia API for summary
    let url = format!(
        "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
        urlencoding::encode(query)
    );

    let response = client
        .get(&url)
        .header("User-Agent", "AgnoRust/1.0 (https://github.com/agno-rust)")
        .send()
        .await
        .map_err(|e| AgnoError::ToolInvocation {
            name: "wikipedia_search".into(),
            source: Box::new(e),
        })?;

    if !response.status().is_success() {
        // Try search API as fallback
        return search_wikipedia_fallback(query).await;
    }

    let json: Value = response.json().await.map_err(|e| AgnoError::ToolInvocation {
        name: "wikipedia_search".into(),
        source: Box::new(e),
    })?;

    let title = json["title"]
        .as_str()
        .unwrap_or(query)
        .to_string();
    let extract = json["extract"]
        .as_str()
        .unwrap_or("No summary available")
        .to_string();

    Ok(WikipediaSummary { title, extract })
}

async fn search_wikipedia_fallback(query: &str) -> Result<WikipediaSummary> {
    let client = reqwest::Client::new();

    // Use search API
    let url = format!(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json&srprop=snippet",
        urlencoding::encode(query)
    );

    let response = client
        .get(&url)
        .header("User-Agent", "AgnoRust/1.0")
        .send()
        .await
        .map_err(|e| AgnoError::ToolInvocation {
            name: "wikipedia_search".into(),
            source: Box::new(e),
        })?;

    let json: Value = response.json().await.map_err(|e| AgnoError::ToolInvocation {
        name: "wikipedia_search".into(),
        source: Box::new(e),
    })?;

    let results = &json["query"]["search"];
    if let Some(first) = results.as_array().and_then(|arr| arr.first()) {
        let title = first["title"].as_str().unwrap_or(query).to_string();
        let snippet = first["snippet"]
            .as_str()
            .unwrap_or("No content available")
            .to_string();
        // Clean HTML tags from snippet
        let extract = snippet
            .replace("<span class=\"searchmatch\">", "")
            .replace("</span>", "");
        Ok(WikipediaSummary { title, extract })
    } else {
        Ok(WikipediaSummary {
            title: query.to_string(),
            extract: "No Wikipedia article found for this query.".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wikipedia_toolkit_creation() {
        let registry = wikipedia_toolkit();
        assert!(registry.get("wikipedia_search").is_some());
    }
}
