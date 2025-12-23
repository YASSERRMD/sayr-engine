# agno-rust

A high-performance Rust implementation of the [agno](https://github.com/agno-agi/agno) AI agent framework. Build production-ready AI agents with multi-provider LLM support, extensive toolkits, and enterprise-grade features.

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

## Features

### Multi-Provider LLM Support (10 Providers)

| Provider | Default Model | Environment Variable |
|----------|---------------|---------------------|
| **OpenAI** | gpt-4 | `OPENAI_API_KEY` |
| **Anthropic** | claude-3-sonnet | `ANTHROPIC_API_KEY` |
| **Google Gemini** | gemini-pro | `GOOGLE_API_KEY` |
| **Cohere** | command-r-plus | `COHERE_API_KEY` |
| **Groq** | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| **Ollama** | llama3.1 | `OLLAMA_HOST` (optional) |
| **Mistral** | mistral-large-latest | `MISTRAL_API_KEY` |
| **Azure OpenAI** | gpt-4 | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` |
| **Together AI** | Llama-3.3-70B-Instruct | `TOGETHER_API_KEY` |
| **Fireworks** | llama-v3p1-70b-instruct | `FIREWORKS_API_KEY` |

### Built-in Toolkits (13 Toolkits)

| Category | Toolkits | Description |
|----------|----------|-------------|
| **Search** | DuckDuckGo, Wikipedia, Arxiv, PubMed | Web, knowledge, and academic search |
| **Communication** | Slack, Gmail, Discord | Messaging and email integration |
| **Development** | GitHub, Shell, HTTP | Code repos, commands, API calls |
| **Data** | SQL, JSON, Calculator | Database queries, data processing |

### Memory and Knowledge

- **Vector Stores**: In-memory, PostgreSQL (pgvector), Qdrant
- **Embedders**: OpenAI, Transformers, Whitespace (testing)
- **Memory Strategies**: Full, Windowed, Summarized, Token-limited
- **Document Chunking**: Sliding window chunker with overlap

### Enterprise Features

- **Guardrails**: PII detection (SSN, credit card, email, phone), prompt injection detection
- **RBAC and Privacy**: Access control, principals, privacy rules
- **Reasoning**: Chain-of-thought orchestration with confidence scoring
- **MCP Support**: Model Context Protocol client with stdio/HTTP transports

### Observability

- **Telemetry**: OpenTelemetry tracing with OTLP export
- **Metrics**: Prometheus exporter for run duration, tool calls, failures
- **Structured Events**: In-memory collector with batch delivery

### Runtime and Deployment

- **HTTP Server**: REST API with SSE streaming
- **Workflows**: Sequential, parallel, and conditional execution
- **Teams**: Multi-agent coordination
- **Config**: File-based or environment variable configuration

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
agno-rust = "0.2"
```

## Quick Start

```rust
use agno_rust::{Agent, OpenAIClient, ToolRegistry};
use agno_rust::tools::{calculator_toolkit, duckduckgo_toolkit};

#[tokio::main]
async fn main() -> agno_rust::Result<()> {
    // Create LLM client
    let model = OpenAIClient::from_env()?.with_model("gpt-4o");
    
    // Register tools
    let mut tools = ToolRegistry::new();
    calculator_toolkit(&mut tools);
    duckduckgo_toolkit(&mut tools, Default::default());
    
    // Create agent
    let mut agent = Agent::new(model).with_tools(tools);
    
    // Chat
    let reply = agent.respond("What is 42 * 17?").await?;
    println!("{reply}");
    
    Ok(())
}
```

## Toolkit Examples

### GitHub Integration

```rust
use agno_rust::tools::{register_github_tools, GitHubClient};

let mut tools = ToolRegistry::new();
register_github_tools(&mut tools); // Uses GITHUB_TOKEN env var

// Tools: github_search_repos, github_get_repo, github_list_issues, github_read_file
```

### Slack Integration

```rust
use agno_rust::tools::{register_slack_tools, SlackClient};

let mut tools = ToolRegistry::new();
register_slack_tools(&mut tools, std::env::var("SLACK_BOT_TOKEN")?);

// Tools: slack_send_message, slack_list_channels, slack_search
```

### SQL Database

```rust
use agno_rust::tools::register_sql_tools;

let mut tools = ToolRegistry::new();
register_sql_tools(&mut tools, "/path/to/database.db");

// Tools: sql_query (read-only by default), sql_schema
```

### Academic Research

```rust
use agno_rust::tools::{register_arxiv_tools, register_pubmed_tools};

let mut tools = ToolRegistry::new();
register_arxiv_tools(&mut tools);   // Search arXiv papers
register_pubmed_tools(&mut tools);  // Search PubMed

// Tools: arxiv_search, pubmed_search
```

## Memory Strategies

```rust
use agno_rust::{WindowedMemoryStrategy, SummarizedMemoryStrategy};

// Keep last 10 messages
let strategy = WindowedMemoryStrategy::new(10);

// Or use summarization for long conversations
let strategy = SummarizedMemoryStrategy::new(5, 5);
```

## Guardrails

```rust
use agno_rust::guardrails::{PiiGuardrail, PromptInjectionGuardrail, GuardrailChain};

let mut chain = GuardrailChain::new();
chain.add(PiiGuardrail::new());
chain.add(PromptInjectionGuardrail::new());

// Check input before sending to LLM
match chain.validate("My SSN is 123-45-6789") {
    GuardrailResult::Block(reason) => println!("Blocked: {}", reason),
    GuardrailResult::Pass => println!("Safe to proceed"),
}
```

## MCP Integration

```rust
use agno_rust::mcp::{McpClient, StdioTransport, McpTools};

// Connect to MCP server
let transport = StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])?;
let client = McpClient::new(transport);
client.initialize().await?;

// Register MCP tools with agent
let mcp_tools = McpTools::new(client);
mcp_tools.register_all(&mut tools).await?;
```

## Observability

```rust
use agno_rust::{init_tracing, init_prometheus_registry};

// Initialize OpenTelemetry tracing
init_tracing("my-agent", Some("http://otel-collector:4317"));

// Initialize Prometheus metrics
init_prometheus_registry();
```

## Architecture

```
agno-rust/
├── src/
│   ├── agent.rs        # Core agent loop
│   ├── llm.rs          # 10 LLM provider clients
│   ├── tools/          # 13 built-in toolkits
│   ├── guardrails.rs   # PII and injection detection
│   ├── memory.rs       # Memory strategies
│   ├── mcp.rs          # MCP client
│   ├── knowledge/      # RAG and vector stores
│   ├── reasoning.rs    # Chain-of-thought
│   └── server.rs       # HTTP runtime
├── cookbook/           # Example agents
└── scripts/            # Development utilities
```

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin

# Format and lint
./scripts/check.sh
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [agno-agi/agno](https://github.com/agno-agi/agno) - the Python agent framework.
