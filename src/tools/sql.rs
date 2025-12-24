#[cfg(feature = "persistence")]
mod inner {
    //! SQL toolkit for executing database queries.
    //!
    //! Provides tools for querying SQLite databases with safety restrictions.

    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::path::PathBuf;

    // ─────────────────────────────────────────────────────────────────────────────
    // SQL Query Tool
    // ─────────────────────────────────────────────────────────────────────────────

    /// Tool for executing SQL queries against a SQLite database
    pub struct SqlQueryTool {
        db_path: PathBuf,
        read_only: bool,
    }

    impl SqlQueryTool {
        pub fn new(db_path: impl Into<PathBuf>) -> Self {
            Self {
                db_path: db_path.into(),
                read_only: true,
            }
        }

        pub fn with_write_access(mut self) -> Self {
            self.read_only = false;
            self
        }

        fn is_safe_query(&self, query: &str) -> bool {
            let query_upper = query.to_uppercase();
            
            // In read-only mode, only allow SELECT queries
            if self.read_only {
                let dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"];
                for keyword in dangerous {
                    if query_upper.contains(keyword) {
                        return false;
                    }
                }
            }

            // Always block dangerous operations
            let always_blocked = ["DROP DATABASE", "DROP TABLE", "TRUNCATE"];
            for keyword in always_blocked {
                if query_upper.contains(keyword) {
                    return false;
                }
            }

            true
        }
    }

    #[async_trait]
    impl Tool for SqlQueryTool {
        fn name(&self) -> &str {
            "sql_query"
        }

        fn description(&self) -> &str {
            "Execute SQL queries against a SQLite database. Returns query results as JSON."
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
                    "message": "Query contains blocked operations. Only SELECT queries are allowed in read-only mode."
                }));
            }

            // Use sqlx to execute the query
            let pool = sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(1)
                .connect(&format!("sqlite:{}", self.db_path.display()))
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Failed to connect to database: {}", e)))?;

            let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(query)
                .fetch_all(&pool)
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Query failed: {}", e)))?;

            // Convert rows to JSON
            use sqlx::{Column, Row};
            let mut results = Vec::new();
            for row in rows {
                let mut row_obj = serde_json::Map::new();
                for (i, col) in row.columns().iter().enumerate() {
                    let name = col.name().to_string();
                    // Try to get as different types
                    if let Ok(val) = row.try_get::<i64, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<f64, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<String, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<bool, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else {
                        row_obj.insert(name, json!(null));
                    }
                }
                results.push(Value::Object(row_obj));
            }

            Ok(json!({
                "query": query,
                "rows": results,
                "row_count": results.len()
            }))
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Schema Inspection Tool
    // ─────────────────────────────────────────────────────────────────────────────

    /// Tool for inspecting database schema
    pub struct SqlSchemaTool {
        db_path: PathBuf,
    }

    impl SqlSchemaTool {
        pub fn new(db_path: impl Into<PathBuf>) -> Self {
            Self {
                db_path: db_path.into(),
            }
        }
    }

    #[async_trait]
    impl Tool for SqlSchemaTool {
        fn name(&self) -> &str {
            "sql_schema"
        }

        fn description(&self) -> &str {
            "Get the schema of a SQLite database including tables and their columns."
        }

        fn parameters(&self) -> Option<Value> {
            Some(json!({
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional specific table to inspect"
                    }
                }
            }))
        }

        async fn call(&self, input: Value) -> crate::Result<Value> {
            let table_name = input["table_name"].as_str();

            let pool = sqlx::sqlite::SqlitePoolOptions::new()
                .max_connections(1)
                .connect(&format!("sqlite:{}", self.db_path.display()))
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Failed to connect to database: {}", e)))?;

            // Get list of tables
            let tables_query = if let Some(table) = table_name {
                format!(
                    "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='{}'",
                    table.replace('\'', "''")
                )
            } else {
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'".to_string()
            };

            let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(&tables_query)
                .fetch_all(&pool)
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Schema query failed: {}", e)))?;

            use sqlx::Row;
            let mut tables = Vec::new();
            for row in rows {
                let name: String = row.try_get("name").unwrap_or_default();
                let sql: String = row.try_get("sql").unwrap_or_default();
                tables.push(json!({
                    "table_name": name,
                    "create_statement": sql
                }));
            }

            Ok(json!({
                "tables": tables,
                "table_count": tables.len()
            }))
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // SQL Toolkit
    // ─────────────────────────────────────────────────────────────────────────────

    use crate::tool::ToolRegistry;

    /// Register SQL tools with a registry for a specific database
    pub fn register_sql_tools(registry: &mut ToolRegistry, db_path: impl Into<PathBuf>) {
        let path: PathBuf = db_path.into();
        registry.register(SqlQueryTool::new(path.clone()));
        registry.register(SqlSchemaTool::new(path));
    }
}

#[cfg(feature = "persistence")]
pub use inner::*;
