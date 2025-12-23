//! Arxiv toolkit for searching academic papers.
//!
//! Provides tools for searching arXiv.org for academic papers and preprints.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Arxiv Search Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for searching arXiv for academic papers
pub struct ArxivSearchTool {
    client: reqwest::Client,
    max_results: usize,
}

impl ArxivSearchTool {
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

impl Default for ArxivSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ArxivSearchTool {
    fn name(&self) -> &str {
        "arxiv_search"
    }

    fn description(&self) -> &str {
        "Search arXiv for academic papers and preprints. Returns titles, authors, abstracts, and links."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for academic papers"
                },
                "category": {
                    "type": "string",
                    "description": "Optional arXiv category (e.g., 'cs.AI', 'physics.hep-th')"
                }
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'query' parameter".into()))?;

        let category = input["category"].as_str();

        // Build the search query
        let search_query = if let Some(cat) = category {
            format!("cat:{} AND all:{}", cat, query)
        } else {
            format!("all:{}", query)
        };

        let url = format!(
            "http://export.arxiv.org/api/query?search_query={}&start=0&max_results={}",
            urlencoding::encode(&search_query),
            self.max_results
        );

        let response = self
            .client
            .get(&url)
            .header("User-Agent", "agno-rust/0.2.0")
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("arXiv request failed: {}", e)))?;

        let xml = response
            .text()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to read response: {}", e)))?;

        // Parse the Atom XML response (simple parsing)
        let mut results = Vec::new();
        
        // Simple XML parsing for entries
        for entry in xml.split("<entry>").skip(1) {
            if let Some(end) = entry.find("</entry>") {
                let entry_xml = &entry[..end];
                
                let title = extract_xml_content(entry_xml, "title")
                    .map(|s| s.replace('\n', " ").trim().to_string());
                let summary = extract_xml_content(entry_xml, "summary")
                    .map(|s| s.replace('\n', " ").trim().to_string());
                let id = extract_xml_content(entry_xml, "id");
                let published = extract_xml_content(entry_xml, "published");
                
                // Extract authors
                let mut authors = Vec::new();
                for author_block in entry_xml.split("<author>").skip(1) {
                    if let Some(name) = extract_xml_content(author_block, "name") {
                        authors.push(name);
                    }
                }

                if title.is_some() {
                    results.push(json!({
                        "title": title,
                        "summary": summary,
                        "authors": authors,
                        "url": id,
                        "published": published
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

fn extract_xml_content(xml: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{}", tag);
    let end_tag = format!("</{}>", tag);
    
    if let Some(start) = xml.find(&start_tag) {
        let after_start = &xml[start..];
        if let Some(content_start) = after_start.find('>') {
            let content_after = &after_start[content_start + 1..];
            if let Some(end) = content_after.find(&end_tag) {
                return Some(content_after[..end].to_string());
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Arxiv Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all Arxiv tools with a registry
pub fn register_arxiv_tools(registry: &mut ToolRegistry) {
    registry.register(ArxivSearchTool::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arxiv_tool_creation() {
        let tool = ArxivSearchTool::new();
        assert_eq!(tool.name(), "arxiv_search");
        assert!(tool.parameters().is_some());
    }

    #[test]
    fn test_xml_extraction() {
        let xml = r#"<title>Test Paper Title</title><author><name>John Doe</name></author>"#;
        assert_eq!(extract_xml_content(xml, "title"), Some("Test Paper Title".to_string()));
        assert_eq!(extract_xml_content(xml, "name"), Some("John Doe".to_string()));
    }
}
