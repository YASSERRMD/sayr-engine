use pyo3::prelude::*;
use std::sync::Arc;
use sayr_engine::{Agent as RustAgent, OpenAIClient, CohereClient, LanguageModel};

// ─────────────────────────────────────────────────────────────────────────────
// Models
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper for OpenAI model configuration
#[pyclass]
#[derive(Clone)]
struct OpenAIChat {
    model_id: String,
    api_key: Option<String>,
}

#[pymethods]
impl OpenAIChat {
    #[new]
    #[pyo3(signature = (id="gpt-4o".to_string(), api_key=None))]
    fn new(id: String, api_key: Option<String>) -> Self {
        Self { model_id: id, api_key }
    }
}

/// Wrapper for Cohere model configuration
#[pyclass]
#[derive(Clone)]
struct CohereChat {
    model_id: String,
    api_key: Option<String>,
}

#[pymethods]
impl CohereChat {
    #[new]
    #[pyo3(signature = (id="command-a-03-2025".to_string(), api_key=None))]
    fn new(id: String, api_key: Option<String>) -> Self {
        Self { model_id: id, api_key }
    }
}

#[derive(FromPyObject)]
enum ModelConfig {
    OpenAI(OpenAIChat),
    Cohere(CohereChat),
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent
// ─────────────────────────────────────────────────────────────────────────────

enum AgentInner {
    OpenAI(RustAgent<OpenAIClient>),
    Cohere(RustAgent<CohereClient>),
}

impl AgentInner {
    async fn respond(&mut self, message: &str) -> Result<String, String> {
        match self {
            Self::OpenAI(agent) => agent.respond(message).await.map_err(|e| e.to_string()),
            Self::Cohere(agent) => agent.respond(message).await.map_err(|e| e.to_string()),
        }
    }
}

/// The main Agent class 
#[pyclass]
struct Agent {
    inner: Arc<tokio::sync::Mutex<AgentInner>>,
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl Agent {
    #[new]
    #[pyo3(signature = (model=None, description=None, markdown=true))]
    fn new(model: Option<ModelConfig>, description: Option<String>, markdown: bool) -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
            
        let inner = match model {
            Some(ModelConfig::Cohere(config)) => {
                let api_key = if let Some(key) = config.api_key {
                    key
                } else {
                    std::env::var("COHERE_API_KEY").map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("COHERE_API_KEY not found"))?
                };
                let client = CohereClient::new(api_key).with_model(config.model_id);
                let mut agent = RustAgent::new(std::sync::Arc::new(client));
                if let Some(desc) = description {
                    agent = agent.with_system_prompt(desc);
                }
                AgentInner::Cohere(agent)
            },
            // Default to OpenAI
            Some(ModelConfig::OpenAI(config)) => {
                 let client = if let Some(key) = config.api_key {
                    OpenAIClient::new(key)
                } else {
                    OpenAIClient::from_env().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                };
                let client = client.with_model(config.model_id);
                let mut agent = RustAgent::new(std::sync::Arc::new(client));
                if let Some(desc) = description {
                    agent = agent.with_system_prompt(desc);
                }
                AgentInner::OpenAI(agent)
            },
            None => {
                 // Default default to OpenAI
                 let config = OpenAIChat { model_id: "gpt-4".to_string(), api_key: None };
                 let client = OpenAIClient::from_env().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                 let client = client.with_model(config.model_id);
                 let mut agent = RustAgent::new(std::sync::Arc::new(client));
                 if let Some(desc) = description {
                    agent = agent.with_system_prompt(desc);
                }
                AgentInner::OpenAI(agent)
            }
        };

        Ok(Agent {
            inner: Arc::new(tokio::sync::Mutex::new(inner)),
            rt,
        })
    }

    /// Get the response from the model and print it
    fn print_response(&self, message: String) -> PyResult<()> {
        let agent = self.inner.clone();
        
        // Block on async execution
        self.rt.block_on(async move {
            let mut agent_lock = agent.lock().await;
            match agent_lock.respond(&message).await {
                Ok(response) => {
                    println!("{}", response);
                    Ok(())
                },
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            }
        })
    }
    
    /// Get the response as a string
    fn run(&self, message: String) -> PyResult<String> {
        let agent = self.inner.clone();
        self.rt.block_on(async move {
            let mut agent_lock = agent.lock().await;
            match agent_lock.respond(&message).await {
                Ok(response) => Ok(response),
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module Definition
// ─────────────────────────────────────────────────────────────────────────────

/// Naive recursive Fibonacci to demonstrate CPU bound performance
#[pyfunction]
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

/// Simulate token counting (CPU bound string processing)
#[pyfunction]
fn calculate_tokens(text: String) -> usize {
    // Naive approximation: split by whitespace and process
    // Rust is efficient at string iteration
    text.split_whitespace().count()
}

#[pymodule]
fn sayr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<OpenAIChat>()?;
    m.add_class::<CohereChat>()?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_tokens, m)?)?;
    Ok(())
}
