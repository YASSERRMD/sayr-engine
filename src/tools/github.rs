//! GitHub toolkit for interacting with GitHub repositories.
//!
//! Provides tools for searching repos, issues, PRs, and reading file contents.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};

// ─────────────────────────────────────────────────────────────────────────────
// GitHub Client
// ─────────────────────────────────────────────────────────────────────────────

/// Shared GitHub API client
#[derive(Clone)]
pub struct GitHubClient {
    http: reqwest::Client,
    token: Option<String>,
    base_url: String,
}

impl GitHubClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            token: std::env::var("GITHUB_TOKEN").ok(),
            base_url: "https://api.github.com".to_string(),
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    async fn get(&self, endpoint: &str) -> crate::Result<Value> {
        let mut request = self
            .http
            .get(format!("{}{}", self.base_url, endpoint))
            .header("User-Agent", "agno-rust/0.2.0")
            .header("Accept", "application/vnd.github.v3+json");

        if let Some(ref token) = self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("GitHub request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(crate::error::AgnoError::Protocol(format!(
                "GitHub API error {}: {}",
                status, body
            )));
        }

        response
            .json()
            .await
            .map_err(|e| crate::error::AgnoError::Protocol(format!("Failed to parse response: {}", e)))
    }
}

impl Default for GitHubClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search Repositories Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for searching GitHub repositories
pub struct GitHubSearchReposTool {
    client: GitHubClient,
}

impl GitHubSearchReposTool {
    pub fn new() -> Self {
        Self {
            client: GitHubClient::new(),
        }
    }

    pub fn with_client(client: GitHubClient) -> Self {
        Self { client }
    }
}

impl Default for GitHubSearchReposTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitHubSearchReposTool {
    fn name(&self) -> &str {
        "github_search_repos"
    }

    fn description(&self) -> &str {
        "Search GitHub repositories by query. Returns repository names, descriptions, stars, and URLs."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for repositories"
                },
                "language": {
                    "type": "string",
                    "description": "Filter by programming language"
                },
                "sort": {
                    "type": "string",
                    "enum": ["stars", "forks", "updated"],
                    "description": "Sort by stars, forks, or recently updated"
                }
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'query' parameter".into()))?;

        let mut search_query = query.to_string();
        
        if let Some(lang) = input["language"].as_str() {
            search_query.push_str(&format!(" language:{}", lang));
        }

        let sort = input["sort"].as_str().unwrap_or("stars");
        
        let endpoint = format!(
            "/search/repositories?q={}&sort={}&per_page=10",
            urlencoding::encode(&search_query),
            sort
        );

        let response = self.client.get(&endpoint).await?;

        let items = response["items"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|repo| {
                        json!({
                            "name": repo["full_name"],
                            "description": repo["description"],
                            "stars": repo["stargazers_count"],
                            "forks": repo["forks_count"],
                            "language": repo["language"],
                            "url": repo["html_url"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "query": query,
            "total_count": response["total_count"],
            "repositories": items
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Get Repository Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for getting repository details
pub struct GitHubGetRepoTool {
    client: GitHubClient,
}

impl GitHubGetRepoTool {
    pub fn new() -> Self {
        Self {
            client: GitHubClient::new(),
        }
    }

    pub fn with_client(client: GitHubClient) -> Self {
        Self { client }
    }
}

impl Default for GitHubGetRepoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitHubGetRepoTool {
    fn name(&self) -> &str {
        "github_get_repo"
    }

    fn description(&self) -> &str {
        "Get detailed information about a GitHub repository."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (user or organization)"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                }
            },
            "required": ["owner", "repo"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let owner = input["owner"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'owner' parameter".into()))?;
        let repo = input["repo"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'repo' parameter".into()))?;

        let endpoint = format!("/repos/{}/{}", owner, repo);
        let response = self.client.get(&endpoint).await?;

        Ok(json!({
            "name": response["full_name"],
            "description": response["description"],
            "stars": response["stargazers_count"],
            "forks": response["forks_count"],
            "open_issues": response["open_issues_count"],
            "language": response["language"],
            "topics": response["topics"],
            "default_branch": response["default_branch"],
            "created_at": response["created_at"],
            "updated_at": response["updated_at"],
            "url": response["html_url"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// List Issues Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for listing repository issues
pub struct GitHubListIssuesTool {
    client: GitHubClient,
}

impl GitHubListIssuesTool {
    pub fn new() -> Self {
        Self {
            client: GitHubClient::new(),
        }
    }

    pub fn with_client(client: GitHubClient) -> Self {
        Self { client }
    }
}

impl Default for GitHubListIssuesTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitHubListIssuesTool {
    fn name(&self) -> &str {
        "github_list_issues"
    }

    fn description(&self) -> &str {
        "List issues from a GitHub repository."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by issue state"
                }
            },
            "required": ["owner", "repo"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let owner = input["owner"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'owner' parameter".into()))?;
        let repo = input["repo"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'repo' parameter".into()))?;
        let state = input["state"].as_str().unwrap_or("open");

        let endpoint = format!("/repos/{}/{}/issues?state={}&per_page=20", owner, repo, state);
        let response = self.client.get(&endpoint).await?;

        let issues = response
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter(|issue| issue["pull_request"].is_null()) // Exclude PRs
                    .map(|issue| {
                        json!({
                            "number": issue["number"],
                            "title": issue["title"],
                            "state": issue["state"],
                            "author": issue["user"]["login"],
                            "labels": issue["labels"].as_array().map(|l| 
                                l.iter().filter_map(|x| x["name"].as_str()).collect::<Vec<_>>()
                            ),
                            "created_at": issue["created_at"],
                            "url": issue["html_url"]
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(json!({
            "repository": format!("{}/{}", owner, repo),
            "state": state,
            "issues": issues,
            "count": issues.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Read File Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for reading file contents from a repository
pub struct GitHubReadFileTool {
    client: GitHubClient,
}

impl GitHubReadFileTool {
    pub fn new() -> Self {
        Self {
            client: GitHubClient::new(),
        }
    }

    pub fn with_client(client: GitHubClient) -> Self {
        Self { client }
    }
}

impl Default for GitHubReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GitHubReadFileTool {
    fn name(&self) -> &str {
        "github_read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file from a GitHub repository."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner"
                },
                "repo": {
                    "type": "string",
                    "description": "Repository name"
                },
                "path": {
                    "type": "string",
                    "description": "Path to the file in the repository"
                },
                "ref": {
                    "type": "string",
                    "description": "Branch, tag, or commit SHA (default: main branch)"
                }
            },
            "required": ["owner", "repo", "path"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let owner = input["owner"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'owner' parameter".into()))?;
        let repo = input["repo"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'repo' parameter".into()))?;
        let path = input["path"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'path' parameter".into()))?;

        let mut endpoint = format!("/repos/{}/{}/contents/{}", owner, repo, path);
        if let Some(git_ref) = input["ref"].as_str() {
            endpoint.push_str(&format!("?ref={}", git_ref));
        }

        let response = self.client.get(&endpoint).await?;

        // Decode base64 content
        let content = response["content"]
            .as_str()
            .map(|c| {
                let cleaned = c.replace('\n', "");
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(&cleaned)
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
            })
            .flatten();

        Ok(json!({
            "path": path,
            "name": response["name"],
            "size": response["size"],
            "sha": response["sha"],
            "content": content,
            "encoding": response["encoding"],
            "url": response["html_url"]
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GitHub Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register all GitHub tools with a registry
pub fn register_github_tools(registry: &mut ToolRegistry) {
    let client = GitHubClient::new();
    registry.register(GitHubSearchReposTool::with_client(client.clone()));
    registry.register(GitHubGetRepoTool::with_client(client.clone()));
    registry.register(GitHubListIssuesTool::with_client(client.clone()));
    registry.register(GitHubReadFileTool::with_client(client));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_github_tools_creation() {
        let search = GitHubSearchReposTool::new();
        assert_eq!(search.name(), "github_search_repos");

        let get_repo = GitHubGetRepoTool::new();
        assert_eq!(get_repo.name(), "github_get_repo");

        let issues = GitHubListIssuesTool::new();
        assert_eq!(issues.name(), "github_list_issues");

        let read_file = GitHubReadFileTool::new();
        assert_eq!(read_file.name(), "github_read_file");
    }
}
