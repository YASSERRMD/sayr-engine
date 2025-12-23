# agno-rust

Rust-native scaffolding inspired by the [agno](https://github.com/agno-agi/agno) agent runtime. The workspace mirrors the structure of the upstream project with core libraries, runnable cookbook examples, and helper scripts.

## Workspace layout

- `libs/agno-core`: Minimal agent loop, registry, transcript memory, and stub language model used for testing.
- `agno-rust` (root crate): Full runtime surface mirroring the upstream Python implementation — governance, telemetry, metrics, deployments, workflows, and a lightweight HTTP runtime.
- `cookbook/echo-agent`: Binary crate demonstrating how to wire tools and a scripted language model into an interactive agent.
- `cookbook/observability-demo`: Shows RBAC, retries, metrics, and telemetry stitched into an agent run.
- `scripts/check.sh`: Convenience script to format the workspace and run the full test suite.

## Feature parity highlights

- **Tooling:** Tools expose names, descriptions, and optional JSON parameter schemas that are embedded into model prompts so the LLM understands expected arguments. Tool descriptions are deterministic and serializable for UIs or remote runtimes.
- **RBAC & privacy:** `AccessController`, `Principal`, and `PrivacyRule` let you block tool calls or redact payloads per tenant.
- **Metrics & telemetry:** `MetricsTracker`, `RetryPolicy`, `FallbackChain`, and `TelemetryCollector` capture reliability signals, failures, and retry attempts.
- **Knowledge & memory:** In-memory vector store, retrievers, and transcript memory let agents ground responses in past context.
- **Workflows & teams:** Build directed workflows, coordinate multiple agents, and expose them over `AgentRuntime`'s HTTP dashboard and SSE event stream.
- **Config & deployment:** Load configs from files or env overrides and emit container-ready `DeploymentPlan` manifests.

## Quickstart (root crate)

```rust
use agno_rust::{Agent, LanguageModel, StubModel, Tool, ToolRegistry};
use async_trait::async_trait;
use serde_json::Value;

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes the input JSON back" }
    fn parameters(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }))
    }

    async fn call(&self, input: Value) -> agno_rust::Result<Value> { Ok(input) }
}

#[tokio::main]
async fn main() -> agno_rust::Result<()> {
    let model = StubModel::new(vec![
        r#"{\"action\":\"call_tool\",\"name\":\"echo\",\"arguments\":{\"text\":\"hi\"}}"#.into(),
        r#"{\"action\":\"respond\",\"content\":\"Echo complete.\"}"#.into(),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(Echo);

    let mut agent = Agent::new(model).with_tools(tools);
    let reply = agent.respond("Please echo hi").await?;

    println!("Agent replied: {reply}");
    Ok(())
}
```

## Cookbook example

To run the `echo-agent` cookbook binary:

```bash
cargo run -p echo-agent
```

The script uses a stub language model to request a tool call, routes the call through the registry, and then emits a final assistant reply.

## Development

Run the formatter and the full test suite across all workspace members:

```bash
./scripts/check.sh
```

If your environment cannot fetch crates from crates.io, the tests may fail when Cargo tries to download dependencies. The core library tests only rely on the stub language model and the in-memory tool wiring.

## Telemetry and metrics

- Initialize tracing with `telemetry::init_tracing(service_name, Some("http://otel-collector:4317"))` to emit OTLP spans. Use `TelemetryLabels` to propagate tenant, tool, and workflow context; `span_with_labels` creates spans with those fields, and `flush_tracer` forces batches out before shutdown.
- Initialize metrics with `metrics::init_prometheus_registry()` to install a Prometheus exporter. `MetricsTracker` records run duration, tool calls, and failures and propagates the same labels to each metric.
- `TelemetryCollector` captures structured events and failures in-memory, while `TelemetrySink` exposes a drainable buffer for batch delivery to downstream systems.
