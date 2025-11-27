
## agno-rust

Rust-native scaffolding inspired by the [agno](https://github.com/agno-agi/agno) agent runtime. The crate focuses on the core event loop: a language model emits structured directives, the agent executes registered tools, and the transcript is fed back for the next turn.

### Features

- `LanguageModel` trait so any completion provider can plug in.
- `Tool` trait with async execution and a registry for discovery.
- `Agent` that enforces a JSON-based protocol (`respond` vs. `call_tool`).
- In-memory transcript via `ConversationMemory`.

### Quickstart

```rust
use agno_rust::{Agent, LanguageModel, StubModel, Tool, ToolRegistry};
use async_trait::async_trait;
use serde_json::Value;

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echoes the input JSON back" }
    async fn call(&self, input: Value) -> agno_rust::Result<Value> { Ok(input) }
}

#[tokio::main]
async fn main() -> agno_rust::Result<()> {
    let model = StubModel::new(vec![
        r#"{"action":"respond","content":"Echo complete."}"#.into(),
        r#"{"action":"call_tool","name":"echo","arguments":{"text":"hi"}}"#.into(),
    ]);

    let mut tools = ToolRegistry::new();
    tools.register(Echo);

    let mut agent = Agent::new(model).with_tools(tools);
    let reply = agent.respond("Please echo hi").await?;

    println!("Agent replied: {reply}");
    Ok(())
}
```

The `StubModel` pops scripted JSON replies, making it easy to test tool routing without a live language model.

