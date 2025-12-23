//! DuckDB toolkit for executing queries.
//!
//! Provides tools for querying DuckDB databases with safety restrictions.

use crate::tool::Tool;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use duckdb::Connection;

// ─────────────────────────────────────────────────────────────────────────────
// DuckDB Query Tool
// ─────────────────────────────────────────────────────────────────────────────

/// Tool for executing queries against a DuckDB database
pub struct DuckDbQueryTool {
    conn: Arc<Mutex<Connection>>,
    read_only: bool,
}

impl DuckDbQueryTool {
    pub fn new_in_memory() -> crate::Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| crate::error::AgnoError::Storage(format!("Failed to open DuckDB: {}", e)))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            read_only: false,
        })
    }

    pub fn open(path: &str) -> crate::Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| crate::error::AgnoError::Storage(format!("Failed to open DuckDB: {}", e)))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            read_only: false,
        })
    }

    pub fn with_read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    fn is_safe_query(&self, query: &str) -> bool {
        let query_upper = query.to_uppercase();
        
        if self.read_only {
            let dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "COPY"];
            for keyword in dangerous {
                if query_upper.contains(keyword) {
                    return false;
                }
            }
        }
        true
    }
}

#[async_trait]
impl Tool for DuckDbQueryTool {
    fn name(&self) -> &str {
        "duckdb_query"
    }

    fn description(&self) -> &str {
        "Execute analytical SQL queries against a DuckDB database."
    }

    fn parameters(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["query"]
        }))
    }

    async fn call(&self, input: Value) -> crate::Result<Value> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| crate::error::AgnoError::Protocol("missing 'query' parameter".into()))?;

        if !self.is_safe_query(query) {
            return Ok(json!({
                "error": true,
                "message": "Query contains blocked operations in read-only mode."
            }));
        }

        let conn = self.conn.lock().map_err(|_| crate::error::AgnoError::Storage("Lock poisoned".into()))?;
        let mut stmt = conn.prepare(query)
            .map_err(|e| crate::error::AgnoError::Storage(format!("Prepare failed: {}", e)))?;

        // DuckDB generic query execution is tricky without specific types.
        // We'll use query_arrow if possible or just execute and try to map basic types.
        // For simplicity in this toolkit, we'll assume JSON output via DuckDB's magic or basic implementation.
        // DuckDB doesn't have a simple "fetch all as JSON" in the rust binding easily exposed without iterating rows.

        // Simpler approach: Map columns manually.
        let column_count = stmt.column_count();
        let column_names: Vec<String> = (0..column_count).map(|i| stmt.column_name(i).map(|s| s.to_string()).unwrap_or_else(|_| "unknown".to_string())).collect();

        let mut rows = stmt.query([])
            .map_err(|e| crate::error::AgnoError::Storage(format!("Query failed: {}", e)))?;

        let mut results = Vec::new();
        while let Some(row) = rows.next().map_err(|e| crate::error::AgnoError::Storage(format!("Row error: {}", e)))? {
            let mut row_map = serde_json::Map::new();
            for (i, name) in column_names.iter().enumerate() {
                // Try basic types
                let val: Value = if let Ok(v) = row.get::<_, i64>(i) {
                    json!(v)
                } else if let Ok(v) = row.get::<_, f64>(i) {
                    json!(v)
                } else if let Ok(v) = row.get::<_, String>(i) {
                    json!(v)
                } else if let Ok(v) = row.get::<_, bool>(i) {
                    json!(v)
                } else {
                    json!(null)
                };
                row_map.insert(name.clone(), val);
            }
            results.push(Value::Object(row_map));
        }

        Ok(json!({
            "query": query,
            "rows": results,
            "row_count": results.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DuckDB Toolkit
// ─────────────────────────────────────────────────────────────────────────────

use crate::tool::ToolRegistry;

/// Register DuckDB tools with a registry
pub fn register_duckdb_tools(registry: &mut ToolRegistry, path: Option<&str>) -> crate::Result<()> {
    let tool = if let Some(p) = path {
        DuckDbQueryTool::open(p)?
    } else {
        DuckDbQueryTool::new_in_memory()?
    };
    registry.register(tool);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duckdb_tool_creation() {
        let tool = DuckDbQueryTool::new_in_memory().unwrap();
        assert_eq!(tool.name(), "duckdb_query");
        assert!(tool.parameters().is_some());
    }
}
