//! PubMed toolkit for searching biomedical literature.
//!
//! Provides tools for searching PubMed/NCBI for medical and life science papers.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// PubMed Search Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for searching PubMed for biomedical literature
pub struct PubmedSearchTool {
    client: reqwest::Client,
    max_results: usize,
}

impl PubmedSearchTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            max_results: 10,
        }
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }
}

impl Default for PubmedSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for PubmedSearchTool {
    fn name(&self) -> &str {
        "pubmed_search"
    }

    fn description(&self) -> &str {
        "Search PubMed for biomedical and life science literature. Returns article titles, authors, abstracts, and PubMed IDs."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for biomedical papers"
                }
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'query' parameter".into()))?;

        // Step 1: Search for IDs using esearch
        let search_url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax={}&retmode=json",
            urlencoding::encode(query),
            self.max_results
        );

        let search_resp = self
            .client
            .get(&search_url)
            .header("User-Agent", "sayr-engine/0.3.0")
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("PubMed search failed: {}", e)))?;

        let search_json: Value = search_resp
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse search response: {}", e)))?;

        let ids: Vec<&str> = search_json["esearchresult"]["idlist"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        if ids.is_empty() {
            return Ok(json!({
                "query": query,
                "results": [],
                "total_results": 0
            }));
        }

        // Step 2: Fetch summaries using esummary
        let ids_str = ids.join(",");
        let summary_url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}&retmode=json",
            ids_str
        );

        let summary_resp = self
            .client
            .get(&summary_url)
            .header("User-Agent", "sayr-engine/0.3.0")
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("PubMed summary failed: {}", e)))?;

        let summary_json: Value = summary_resp
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse summary response: {}", e)))?;

        let mut results = Vec::new();
        
        if let Some(result_obj) = summary_json["result"].as_object() {
            for id in &ids {
                if let Some(article) = result_obj.get(*id) {
                    let title = article["title"].as_str().unwrap_or("");
                    let pub_date = article["pubdate"].as_str().unwrap_or("");
                    let source = article["source"].as_str().unwrap_or("");
                    
                    let authors: Vec<String> = article["authors"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|a| a["name"].as_str())
                                .map(String::from)
                                .collect()
                        })
                        .unwrap_or_default();

                    results.push(json!({
                        "pmid": id,
                        "title": title,
                        "authors": authors,
                        "journal": source,
                        "pub_date": pub_date,
                        "url": format!("https://pubmed.ncbi.nlm.nih.gov/{}/", id)
                    }));
                }
            }
        }

        Ok(json!({
            "query": query,
            "results": results,
            "total_results": results.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PubMed Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all PubMed tools with a registry
pub fn register_pubmed_tools(registry: &mut ToolRegistry) {
    registry.register(PubmedSearchTool::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pubmed_tool_creation() {
        let tool = PubmedSearchTool::new();
        assert_eq!(tool.name(), "pubmed_search");
        assert!(tool.parameters().is_some());
    }
}
