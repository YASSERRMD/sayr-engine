#[cfg(feature = "persistence")]
mod inner {
    //! PostgreSQL toolkit for executing queries.
    //!
    //! Provides tools for querying PostgreSQL databases with safety restrictions.

    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::{json, Value};

    // ─────────────────────────────────────────────────────────────────────────────
    // Postgres Query Tool
    // ─────────────────────────────────────────────────────────────────────────────

    /// Tool for executing queries against a PostgreSQL database
    pub struct PostgresQueryTool {
        connection_string: String,
        read_only: bool,
    }

    impl PostgresQueryTool {
        pub fn new(connection_string: impl Into<String>) -> Self {
            Self {
                connection_string: connection_string.into(),
                read_only: true,
            }
        }

        pub fn with_write_access(mut self) -> Self {
            self.read_only = false;
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
    impl Tool for PostgresQueryTool {
        fn name(&self) -> &str {
            "postgres_query"
        }

        fn description(&self) -> &str {
            "Execute SQL queries against a PostgreSQL database."
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

            let pool = sqlx::postgres::PgPoolOptions::new()
                .max_connections(1)
                .connect(&self.connection_string)
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Failed to connect to Postgres: {}", e)))?;

            let rows = sqlx::query(query)
                .fetch_all(&pool)
                .await
                .map_err(|e| crate::error::AgnoError::Storage(format!("Query failed: {}", e)))?;

            use sqlx::{Column, Row};
            let mut results = Vec::new();
            for row in rows {
                let mut row_obj = serde_json::Map::new();
                for (i, col) in row.columns().iter().enumerate() {
                    let name = col.name().to_string();
                    
                    // Try different types common in Postgres
                    if let Ok(val) = row.try_get::<i64, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<i32, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<f64, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<String, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(val) = row.try_get::<bool, _>(i) {
                        row_obj.insert(name, json!(val));
                    } else if let Ok(_val) = row.try_get::<uuid::Uuid, _>(i) {
                        row_obj.insert(name, json!(_val.to_string()));
                    //} else if let Ok(_val) = row.try_get::<serde_json::Value, _>(i) {
                    //    row_obj.insert(name, _val);
                    } else {
                        // Fallback to string for unknown types or just null
                        // For now null to be safe from panics if try_get fails 
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
    // Postgres Toolkit
    // ─────────────────────────────────────────────────────────────────────────────

    use crate::tool::ToolRegistry;

    /// Register Postgres tools with a registry
    pub fn register_postgres_tools(registry: &mut ToolRegistry, connection_string: impl Into<String>) {
        registry.register(PostgresQueryTool::new(connection_string));
    }
}

#[cfg(feature = "persistence")]
pub use inner::*;
