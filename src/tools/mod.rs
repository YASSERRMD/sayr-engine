//! Tools module - provides various toolkits for agents.
//!
//! This module contains implementations of common tools that agents can use:
//! - Calculator: Math operations
//! - DuckDuckGo: Web search
//! - Shell: Command execution
//! - HTTP: API requests
//! - Wikipedia: Knowledge search
//! - JSON: Data parsing and validation

pub mod calculator;
pub mod duckduckgo;
pub mod http;
pub mod json;
pub mod shell;
pub mod wikipedia;

pub use calculator::calculator_toolkit;
pub use duckduckgo::{duckduckgo_toolkit, DuckDuckGoConfig, SearchResult};
pub use http::{http_api_toolkit, HttpApiConfig};
pub use json::json_toolkit;
pub use shell::{shell_toolkit, ShellConfig};
pub use wikipedia::wikipedia_toolkit;
