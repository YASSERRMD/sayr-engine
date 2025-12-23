//! Tools module - provides various toolkits for agents.
//!
//! This module contains implementations of common tools that agents can use:
//! - Calculator: Math operations
//! - DuckDuckGo: Web search
//! - Shell: Command execution
//! - HTTP: API requests
//! - Wikipedia: Knowledge search
//! - JSON: Data parsing and validation
//! - Arxiv: Academic paper search
//! - PubMed: Biomedical literature search
//! - SQL: Database queries
//! - GitHub: Repository and code search
//! - Slack: Messaging
//! - Gmail: Email
//! - Discord: Chat

pub mod arxiv;
pub mod calculator;
pub mod discord;
pub mod duckduckgo;
pub mod github;
pub mod gmail;
pub mod http;
pub mod json;
pub mod pubmed;
pub mod shell;
pub mod slack;
pub mod sql;
pub mod wikipedia;

pub use arxiv::{register_arxiv_tools, ArxivSearchTool};
pub use calculator::calculator_toolkit;
pub use discord::{register_discord_tools, DiscordClient};
pub use duckduckgo::{duckduckgo_toolkit, DuckDuckGoConfig, SearchResult};
pub use github::{register_github_tools, GitHubClient};
pub use gmail::{register_gmail_tools, GmailClient};
pub use http::{http_api_toolkit, HttpApiConfig};
pub use json::json_toolkit;
pub use pubmed::{register_pubmed_tools, PubmedSearchTool};
pub use shell::{shell_toolkit, ShellConfig};
pub use slack::{register_slack_tools, SlackClient};
pub use sql::{register_sql_tools, SqlQueryTool, SqlSchemaTool};
pub use wikipedia::wikipedia_toolkit;
