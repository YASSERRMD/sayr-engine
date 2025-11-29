# agno-rust

Rust-native scaffolding inspired by the [agno](https://github.com/agno-agi/agno) agent runtime. The workspace mirrors the structure of the upstream project with core libraries, runnable cookbook examples, and helper scripts.

## Workspace layout

- `libs/agno-core`: Core library that exposes the agent loop, tool registry, transcript memory, and stub language model used for testing. Tools can optionally publish JSON argument schemas that are embedded in prompts.
- `cookbook/echo-agent`: A small binary crate demonstrating how to wire tools and a scripted language model into an interactive agent.
- `scripts/check.sh`: Convenience script to format the workspace and run the full test suite.

## agno-core quickstart

```rust
use agno_core::{Agent, LanguageModel, StubModel, Tool, ToolRegistry};
use async_trait::async_trait;
use serde_json::Value;

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes the input JSON back" }
    async fn call(&self, input: Value) -> agno_core::Result<Value> { Ok(input) }
}

#[tokio::main]
async fn main() -> agno_core::Result<()> {
    let model = StubModel::new(vec![
        r#"{"action":"call_tool","name":"echo","arguments":{"text":"hi"}}"#.into(),
        r#"{"action":"respond","content":"Echo complete."}"#.into(),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(Echo);

    let mut agent = Agent::new(model).with_tools(tools);
    let reply = agent.respond("Please echo hi").await?;

    println!("Agent replied: {reply}");
    Ok(())
}
```

## Operational tooling

- **RBAC & governance:** `AccessController`, `Principal`, and privacy rules let you block tool calls or redact payloads per tenant.
- **Metrics:** `MetricsTracker` captures duration, memory, tool-call counts, and success/failure ratios to audit agent reliability.
- **Telemetry:** `TelemetryCollector`, `RetryPolicy`, and `FallbackChain` record failures while adding automatic retries and fallbacks for brittle integrations.
- **Config & deployment:** Load `AppConfig` from files or environment overrides and render container-ready manifests with `DeploymentPlan`.
- **Cookbook:** Check `cookbook/observability-demo` and `cookbook/templates/app-config.toml` for ready-to-run examples.

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
