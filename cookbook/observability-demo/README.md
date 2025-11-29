# Observability and Reliability Demo

This walkthrough shows how to wire telemetry, metrics, retries, and RBAC into an agent.

```rust
use agno_rust::{
    AccessController, Action, Agent, GovernanceRole, MetricsTracker, Principal, RetryPolicy,
    StubModel, TelemetryCollector, Tool, ToolRegistry,
};
use async_trait::async_trait;
use serde_json::Value;

struct HealthCheck;

#[async_trait]
impl Tool for HealthCheck {
    fn name(&self) -> &str { "health_check" }
    fn description(&self) -> &str { "Returns ok when the system is healthy" }
    async fn call(&self, _input: Value) -> agno_rust::Result<Value> {
        Ok(serde_json::json!({"status": "ok"}))
    }
}

#[tokio::main]
async fn main() -> agno_rust::Result<()> {
    let mut registry = ToolRegistry::new();
    registry.register(HealthCheck);

    let metrics = MetricsTracker::default();
    let telemetry = TelemetryCollector::default();

    let mut acl = AccessController::new();
    acl.allow(GovernanceRole::User, Action::CallTool("health_check".into()));

    let principal = Principal { id: "demo-user".into(), role: GovernanceRole::User, tenant: Some("acme".into()) };

    let model = StubModel::new(vec![r#"{"action":"call_tool","name":"health_check","arguments":{}}"#.into(), r#"{"action":"respond","content":"All clear"}"#.into()]);
    let mut agent = Agent::new(model)
        .with_tools(registry)
        .with_metrics(metrics.clone())
        .with_telemetry(telemetry.clone())
        .with_access_control(std::sync::Arc::new(acl))
        .with_principal(principal);

    let policy = RetryPolicy::default_external_call();
    let reply = policy.retry(|_| agent.respond("run checks"), Some(&telemetry)).await?;
    println!("reply: {reply}");
    println!("run reports: {}", metrics.reports().len());
    Ok(())
}
```
