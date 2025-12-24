use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;

use sayr_engine::{
    basic_toolkit, Agent as RustAgent, AppConfig, Attachment, AttachmentKind, DeploymentConfig,
    Message, ModelConfig, OpenAIClient, ProviderConfig, Role, SecurityConfig, ServerConfig,
    TelemetryConfig, TelemetryLabels, ToolCall, ToolDescription, ToolRegistry, ToolResult,
    CohereClient,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn not_implemented_error(name: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "{name} is not yet bound for Python usage."
    ))
}

macro_rules! stub_pyclass {
    ($struct_name:ident, $py_name:literal) => {
        #[pyclass(name = $py_name)]
        struct $struct_name;

        #[pymethods]
        impl $struct_name {
            #[new]
            fn new() -> PyResult<Self> {
                Err(not_implemented_error($py_name))
            }
        }
    };
}

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
    #[pyo3(signature = (id=None, api_key=None))]
    fn new(id: Option<String>, api_key: Option<String>) -> Self {
        Self {
            model_id: id.unwrap_or_else(|| "gpt-4o".to_string()),
            api_key,
        }
    }

    #[getter]
    fn id(&self) -> String {
        self.model_id.clone()
    }

    #[setter]
    fn set_id(&mut self, id: String) {
        self.model_id = id;
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.api_key.clone()
    }

    #[setter]
    fn set_api_key(&mut self, api_key: Option<String>) {
        self.api_key = api_key;
    }
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
    #[pyo3(signature = (model=None, description=None, _markdown=true))]
    fn new(
        model: Option<PyModelConfig>,
        description: Option<String>,
        _markdown: bool,
    ) -> PyResult<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        // Default to OpenAI if no model config provided
        let model_config = model.unwrap_or_else(|| PyModelConfig::new(
            "openai".to_string(),
            "gpt-4".to_string(),
            None, None, None, false, None, None, None, None
        ));

        let inner = match model_config.provider().as_str() {
            "cohere" => {
                let cohere_config = model_config.cohere();
                let api_key = cohere_config.api_key().or(model_config.api_key()).or(std::env::var("COHERE_API_KEY").ok())
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("COHERE_API_KEY not found"))?;
                
                let client = CohereClient::new(api_key).with_model(model_config.model());
                let mut agent = RustAgent::new(std::sync::Arc::new(client));
                if let Some(desc) = description {
                    agent = agent.with_system_prompt(desc);
                }
                AgentInner::Cohere(agent)
            },
            "openai" | _ => {
                let openai_config = model_config.openai();
                let client = if let Some(key) = openai_config.api_key().or(model_config.api_key()) {
                    OpenAIClient::new(key)
                } else {
                    OpenAIClient::from_env()
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                };
                let client = client.with_model(model_config.model());
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

        self.rt.block_on(async move {
            let mut agent_lock = agent.lock().await;
            match agent_lock.respond(&message).await {
                Ok(response) => {
                    println!("{}", response);
                    Ok(())
                }
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e,
                )),
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
                Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e,
                )),
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config bindings
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "ServerConfig")]
#[derive(Clone)]
struct PyServerConfig {
    inner: ServerConfig,
}

#[pymethods]
impl PyServerConfig {
    #[new]
    #[pyo3(signature = (host="0.0.0.0".to_string(), port=8080, tls_enabled=false))]
    fn new(host: String, port: u16, tls_enabled: bool) -> Self {
        Self {
            inner: ServerConfig {
                host,
                port,
                tls_enabled,
            },
        }
    }

    #[getter]
    fn host(&self) -> String {
        self.inner.host.clone()
    }

    #[setter]
    fn set_host(&mut self, host: String) {
        self.inner.host = host;
    }

    #[getter]
    fn port(&self) -> u16 {
        self.inner.port
    }

    #[setter]
    fn set_port(&mut self, port: u16) {
        self.inner.port = port;
    }

    #[getter]
    fn tls_enabled(&self) -> bool {
        self.inner.tls_enabled
    }

    #[setter]
    fn set_tls_enabled(&mut self, tls_enabled: bool) {
        self.inner.tls_enabled = tls_enabled;
    }
}

#[pyclass(name = "SecurityConfig")]
#[derive(Clone)]
struct PySecurityConfig {
    inner: SecurityConfig,
}

#[pymethods]
impl PySecurityConfig {
    #[new]
    #[pyo3(signature = (allowed_origins=None, allowed_tenants=None, encryption_required=true))]
    fn new(
        allowed_origins: Option<Vec<String>>,
        allowed_tenants: Option<Vec<String>>,
        encryption_required: bool,
    ) -> Self {
        Self {
            inner: SecurityConfig {
                allowed_origins: allowed_origins.unwrap_or_default(),
                allowed_tenants: allowed_tenants.unwrap_or_default(),
                encryption_required,
            },
        }
    }

    #[getter]
    fn allowed_origins(&self) -> Vec<String> {
        self.inner.allowed_origins.clone()
    }

    #[setter]
    fn set_allowed_origins(&mut self, allowed_origins: Vec<String>) {
        self.inner.allowed_origins = allowed_origins;
    }

    #[getter]
    fn allowed_tenants(&self) -> Vec<String> {
        self.inner.allowed_tenants.clone()
    }

    #[setter]
    fn set_allowed_tenants(&mut self, allowed_tenants: Vec<String>) {
        self.inner.allowed_tenants = allowed_tenants;
    }

    #[getter]
    fn encryption_required(&self) -> bool {
        self.inner.encryption_required
    }

    #[setter]
    fn set_encryption_required(&mut self, encryption_required: bool) {
        self.inner.encryption_required = encryption_required;
    }
}

#[pyclass(name = "TelemetryConfig")]
#[derive(Clone)]
struct PyTelemetryConfig {
    inner: TelemetryConfig,
}

#[pymethods]
impl PyTelemetryConfig {
    #[new]
    #[pyo3(signature = (sample_rate=1.0, endpoint=None, retention_hours=72))]
    fn new(sample_rate: f32, endpoint: Option<String>, retention_hours: u32) -> Self {
        Self {
            inner: TelemetryConfig {
                sample_rate,
                endpoint,
                retention_hours,
            },
        }
    }

    #[getter]
    fn sample_rate(&self) -> f32 {
        self.inner.sample_rate
    }

    #[setter]
    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.inner.sample_rate = sample_rate;
    }

    #[getter]
    fn endpoint(&self) -> Option<String> {
        self.inner.endpoint.clone()
    }

    #[setter]
    fn set_endpoint(&mut self, endpoint: Option<String>) {
        self.inner.endpoint = endpoint;
    }

    #[getter]
    fn retention_hours(&self) -> u32 {
        self.inner.retention_hours
    }

    #[setter]
    fn set_retention_hours(&mut self, retention_hours: u32) {
        self.inner.retention_hours = retention_hours;
    }
}

#[pyclass(name = "DeploymentConfig")]
#[derive(Clone)]
struct PyDeploymentConfig {
    inner: DeploymentConfig,
}

#[pymethods]
impl PyDeploymentConfig {
    #[new]
    #[pyo3(signature = (replicas=1, max_concurrency=32, autoscale=false, container_image=None))]
    fn new(
        replicas: u16,
        max_concurrency: u32,
        autoscale: bool,
        container_image: Option<String>,
    ) -> Self {
        Self {
            inner: DeploymentConfig {
                replicas,
                max_concurrency,
                autoscale,
                container_image,
            },
        }
    }

    #[getter]
    fn replicas(&self) -> u16 {
        self.inner.replicas
    }

    #[setter]
    fn set_replicas(&mut self, replicas: u16) {
        self.inner.replicas = replicas;
    }

    #[getter]
    fn max_concurrency(&self) -> u32 {
        self.inner.max_concurrency
    }

    #[setter]
    fn set_max_concurrency(&mut self, max_concurrency: u32) {
        self.inner.max_concurrency = max_concurrency;
    }

    #[getter]
    fn autoscale(&self) -> bool {
        self.inner.autoscale
    }

    #[setter]
    fn set_autoscale(&mut self, autoscale: bool) {
        self.inner.autoscale = autoscale;
    }

    #[getter]
    fn container_image(&self) -> Option<String> {
        self.inner.container_image.clone()
    }

    #[setter]
    fn set_container_image(&mut self, container_image: Option<String>) {
        self.inner.container_image = container_image;
    }
}

#[pyclass(name = "ProviderConfig")]
#[derive(Clone)]
struct PyProviderConfig {
    inner: ProviderConfig,
}

#[pymethods]
impl PyProviderConfig {
    #[new]
    #[pyo3(signature = (api_key=None, endpoint=None, organization=None))]
    fn new(
        api_key: Option<String>,
        endpoint: Option<String>,
        organization: Option<String>,
    ) -> Self {
        Self {
            inner: ProviderConfig {
                api_key,
                endpoint,
                organization,
            },
        }
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }

    #[setter]
    fn set_api_key(&mut self, api_key: Option<String>) {
        self.inner.api_key = api_key;
    }

    #[getter]
    fn endpoint(&self) -> Option<String> {
        self.inner.endpoint.clone()
    }

    #[setter]
    fn set_endpoint(&mut self, endpoint: Option<String>) {
        self.inner.endpoint = endpoint;
    }

    #[getter]
    fn organization(&self) -> Option<String> {
        self.inner.organization.clone()
    }

    #[setter]
    fn set_organization(&mut self, organization: Option<String>) {
        self.inner.organization = organization;
    }
}

#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
struct PyModelConfig {
    inner: ModelConfig,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    #[pyo3(signature = (provider="stub".to_string(), model="stub-model".to_string(), api_key=None, base_url=None, organization=None, stream=false, openai=None, anthropic=None, gemini=None, cohere=None))]
    fn new(
        provider: String,
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
        organization: Option<String>,
        stream: bool,
        openai: Option<PyProviderConfig>,
        anthropic: Option<PyProviderConfig>,
        gemini: Option<PyProviderConfig>,
        cohere: Option<PyProviderConfig>,
    ) -> Self {
        Self {
            inner: ModelConfig {
                provider,
                model,
                api_key,
                base_url,
                organization,
                stream,
                openai: openai.map(|p| p.inner).unwrap_or_default(),
                anthropic: anthropic.map(|p| p.inner).unwrap_or_default(),
                gemini: gemini.map(|p| p.inner).unwrap_or_default(),
                cohere: cohere.map(|p| p.inner).unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn provider(&self) -> String {
        self.inner.provider.clone()
    }

    #[setter]
    fn set_provider(&mut self, provider: String) {
        self.inner.provider = provider;
    }

    #[getter]
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    #[setter]
    fn set_model(&mut self, model: String) {
        self.inner.model = model;
    }

    #[getter]
    fn api_key(&self) -> Option<String> {
        self.inner.api_key.clone()
    }

    #[setter]
    fn set_api_key(&mut self, api_key: Option<String>) {
        self.inner.api_key = api_key;
    }

    #[getter]
    fn base_url(&self) -> Option<String> {
        self.inner.base_url.clone()
    }

    #[setter]
    fn set_base_url(&mut self, base_url: Option<String>) {
        self.inner.base_url = base_url;
    }

    #[getter]
    fn organization(&self) -> Option<String> {
        self.inner.organization.clone()
    }

    #[setter]
    fn set_organization(&mut self, organization: Option<String>) {
        self.inner.organization = organization;
    }

    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }

    #[setter]
    fn set_stream(&mut self, stream: bool) {
        self.inner.stream = stream;
    }

    #[getter]
    fn openai(&self) -> PyProviderConfig {
        PyProviderConfig {
            inner: self.inner.openai.clone(),
        }
    }

    #[setter]
    fn set_openai(&mut self, openai: PyProviderConfig) {
        self.inner.openai = openai.inner;
    }

    #[getter]
    fn anthropic(&self) -> PyProviderConfig {
        PyProviderConfig {
            inner: self.inner.anthropic.clone(),
        }
    }

    #[setter]
    fn set_anthropic(&mut self, anthropic: PyProviderConfig) {
        self.inner.anthropic = anthropic.inner;
    }

    #[getter]
    fn gemini(&self) -> PyProviderConfig {
        PyProviderConfig {
            inner: self.inner.gemini.clone(),
        }
    }

    #[setter]
    fn set_gemini(&mut self, gemini: PyProviderConfig) {
        self.inner.gemini = gemini.inner;
    }

    #[getter]
    fn cohere(&self) -> PyProviderConfig {
        PyProviderConfig {
            inner: self.inner.cohere.clone(),
        }
    }

    #[setter]
    fn set_cohere(&mut self, cohere: PyProviderConfig) {
        self.inner.cohere = cohere.inner;
    }
}

#[pyclass(name = "AppConfig")]
#[derive(Clone)]
struct PyAppConfig {
    inner: AppConfig,
}

#[pymethods]
impl PyAppConfig {
    #[new]
    #[pyo3(signature = (server, model, security=None, telemetry=None, deployment=None))]
    fn new(
        server: PyServerConfig,
        model: PyModelConfig,
        security: Option<PySecurityConfig>,
        telemetry: Option<PyTelemetryConfig>,
        deployment: Option<PyDeploymentConfig>,
    ) -> Self {
        let mut config = AppConfig::default();
        config.server = server.inner;
        config.security = security.map(|s| s.inner).unwrap_or_default();
        config.telemetry = telemetry.map(|t| t.inner).unwrap_or_default();
        config.deployment = deployment.map(|d| d.inner).unwrap_or_default();
        config.model = model.inner;
        Self { inner: config }
    }

    #[classmethod]
    fn from_file(_cls: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let cfg = AppConfig::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: cfg })
    }

    #[getter]
    fn server(&self) -> PyServerConfig {
        PyServerConfig {
            inner: self.inner.server.clone(),
        }
    }

    #[setter]
    fn set_server(&mut self, server: PyServerConfig) {
        self.inner.server = server.inner;
    }

    #[getter]
    fn security(&self) -> PySecurityConfig {
        PySecurityConfig {
            inner: self.inner.security.clone(),
        }
    }

    #[setter]
    fn set_security(&mut self, security: PySecurityConfig) {
        self.inner.security = security.inner;
    }

    #[getter]
    fn telemetry(&self) -> PyTelemetryConfig {
        PyTelemetryConfig {
            inner: self.inner.telemetry.clone(),
        }
    }

    #[setter]
    fn set_telemetry(&mut self, telemetry: PyTelemetryConfig) {
        self.inner.telemetry = telemetry.inner;
    }

    #[getter]
    fn deployment(&self) -> PyDeploymentConfig {
        PyDeploymentConfig {
            inner: self.inner.deployment.clone(),
        }
    }

    #[setter]
    fn set_deployment(&mut self, deployment: PyDeploymentConfig) {
        self.inner.deployment = deployment.inner;
    }

    #[getter]
    fn model(&self) -> PyModelConfig {
        PyModelConfig {
            inner: self.inner.model.clone(),
        }
    }

    #[setter]
    fn set_model(&mut self, model: PyModelConfig) {
        self.inner.model = model.inner;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message bindings
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "AttachmentKind")]
#[derive(Clone)]
struct PyAttachmentKind {
    inner: AttachmentKind,
}

#[pymethods]
impl PyAttachmentKind {
    #[classmethod]
    fn file(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: AttachmentKind::File,
        }
    }

    #[classmethod]
    fn image(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: AttachmentKind::Image,
        }
    }

    #[classmethod]
    fn audio(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: AttachmentKind::Audio,
        }
    }

    #[classmethod]
    fn video(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: AttachmentKind::Video,
        }
    }

    #[classmethod]
    fn other(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: AttachmentKind::Other,
        }
    }

    fn value(&self) -> &'static str {
        match self.inner {
            AttachmentKind::File => "file",
            AttachmentKind::Image => "image",
            AttachmentKind::Audio => "audio",
            AttachmentKind::Video => "video",
            AttachmentKind::Other => "other",
        }
    }
}

#[pyclass(name = "Role")]
#[derive(Clone)]
struct PyRole {
    inner: Role,
}

#[pymethods]
impl PyRole {
    #[classmethod]
    fn system(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Role::System,
        }
    }

    #[classmethod]
    fn user(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: Role::User }
    }

    #[classmethod]
    fn assistant(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Role::Assistant,
        }
    }

    #[classmethod]
    fn tool(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: Role::Tool }
    }

    fn value(&self) -> &'static str {
        match self.inner {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[pyclass(name = "Attachment")]
#[derive(Clone)]
struct PyAttachment {
    inner: Attachment,
}

#[pymethods]
impl PyAttachment {
    #[new]
    #[pyo3(signature = (kind, uri, description=None, media_type=None))]
    fn new(
        kind: PyAttachmentKind,
        uri: String,
        description: Option<String>,
        media_type: Option<String>,
    ) -> Self {
        Self {
            inner: Attachment {
                kind: kind.inner,
                uri,
                description,
                media_type,
            },
        }
    }

    #[getter]
    fn kind(&self) -> PyAttachmentKind {
        PyAttachmentKind {
            inner: self.inner.kind.clone(),
        }
    }

    #[getter]
    fn uri(&self) -> String {
        self.inner.uri.clone()
    }

    #[setter]
    fn set_uri(&mut self, uri: String) {
        self.inner.uri = uri;
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    #[setter]
    fn set_description(&mut self, description: Option<String>) {
        self.inner.description = description;
    }

    #[getter]
    fn media_type(&self) -> Option<String> {
        self.inner.media_type.clone()
    }

    #[setter]
    fn set_media_type(&mut self, media_type: Option<String>) {
        self.inner.media_type = media_type;
    }
}

#[pyclass(name = "ToolCall")]
#[derive(Clone)]
struct PyToolCall {
    inner: ToolCall,
}

#[pymethods]
impl PyToolCall {
    #[new]
    #[pyo3(signature = (name, arguments_json, id=None))]
    fn new(name: String, arguments_json: String, id: Option<String>) -> PyResult<Self> {
        let arguments = serde_json::from_str(&arguments_json)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self {
            inner: ToolCall {
                id,
                name,
                arguments,
            },
        })
    }

    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[setter]
    fn set_id(&mut self, id: Option<String>) {
        self.inner.id = id;
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.name = name;
    }

    fn arguments_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.arguments)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pyclass(name = "ToolResult")]
#[derive(Clone)]
struct PyToolResult {
    inner: ToolResult,
}

#[pymethods]
impl PyToolResult {
    #[new]
    #[pyo3(signature = (name, output_json, tool_call_id=None))]
    fn new(name: String, output_json: String, tool_call_id: Option<String>) -> PyResult<Self> {
        let output = serde_json::from_str(&output_json)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self {
            inner: ToolResult {
                name,
                output,
                tool_call_id,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.name = name;
    }

    #[getter]
    fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    #[setter]
    fn set_tool_call_id(&mut self, tool_call_id: Option<String>) {
        self.inner.tool_call_id = tool_call_id;
    }

    fn output_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.output)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pyclass(name = "Message")]
#[derive(Clone)]
struct PyMessage {
    inner: Message,
}

#[pymethods]
impl PyMessage {
    #[new]
    #[pyo3(signature = (role, content, tool_call=None, tool_result=None, attachments=None))]
    fn new(
        role: PyRole,
        content: String,
        tool_call: Option<PyToolCall>,
        tool_result: Option<PyToolResult>,
        attachments: Option<Vec<PyAttachment>>,
    ) -> Self {
        Self {
            inner: Message {
                role: role.inner,
                content,
                tool_call: tool_call.map(|t| t.inner),
                tool_result: tool_result.map(|t| t.inner),
                attachments: attachments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|a| a.inner)
                    .collect(),
            },
        }
    }

    #[classmethod]
    fn system(_cls: &Bound<'_, PyType>, content: String) -> Self {
        Self {
            inner: Message::system(content),
        }
    }

    #[classmethod]
    fn user(_cls: &Bound<'_, PyType>, content: String) -> Self {
        Self {
            inner: Message::user(content),
        }
    }

    #[classmethod]
    fn assistant(_cls: &Bound<'_, PyType>, content: String) -> Self {
        Self {
            inner: Message::assistant(content),
        }
    }

    #[classmethod]
    #[pyo3(signature = (name, output_json))]
    fn tool(_cls: &Bound<'_, PyType>, name: String, output_json: String) -> PyResult<Self> {
        let output = serde_json::from_str(&output_json)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Self {
            inner: Message::tool(name, output),
        })
    }

    #[getter]
    fn role(&self) -> PyRole {
        PyRole {
            inner: self.inner.role.clone(),
        }
    }

    #[getter]
    fn content(&self) -> String {
        self.inner.content.clone()
    }

    #[setter]
    fn set_content(&mut self, content: String) {
        self.inner.content = content;
    }

    #[getter]
    fn tool_call(&self) -> Option<PyToolCall> {
        self.inner
            .tool_call
            .clone()
            .map(|tool_call| PyToolCall { inner: tool_call })
    }

    #[setter]
    fn set_tool_call(&mut self, tool_call: Option<PyToolCall>) {
        self.inner.tool_call = tool_call.map(|t| t.inner);
    }

    #[getter]
    fn tool_result(&self) -> Option<PyToolResult> {
        self.inner
            .tool_result
            .clone()
            .map(|tool_result| PyToolResult { inner: tool_result })
    }

    #[setter]
    fn set_tool_result(&mut self, tool_result: Option<PyToolResult>) {
        self.inner.tool_result = tool_result.map(|t| t.inner);
    }

    #[getter]
    fn attachments(&self) -> Vec<PyAttachment> {
        self.inner
            .attachments
            .clone()
            .into_iter()
            .map(|a| PyAttachment { inner: a })
            .collect()
    }

    #[setter]
    fn set_attachments(&mut self, attachments: Vec<PyAttachment>) {
        self.inner.attachments = attachments.into_iter().map(|a| a.inner).collect();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool bindings
// ─────────────────────────────────────────────────────────────────────────────

stub_pyclass!(PyTool, "Tool");

#[pyclass(name = "ToolDescription")]
#[derive(Clone)]
struct PyToolDescription {
    inner: ToolDescription,
}

#[pymethods]
impl PyToolDescription {
    #[new]
    #[pyo3(signature = (name, description, parameters_json=None))]
    fn new(name: String, description: String, parameters_json: Option<String>) -> PyResult<Self> {
        let parameters =
            match parameters_json {
                Some(value) => Some(serde_json::from_str(&value).map_err(|err| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
                })?),
                None => None,
            };
        Ok(Self {
            inner: ToolDescription {
                name,
                description,
                parameters,
            },
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    fn parameters_json(&self) -> PyResult<Option<String>> {
        match &self.inner.parameters {
            Some(value) => serde_json::to_string(value)
                .map(Some)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())),
            None => Ok(None),
        }
    }
}

#[pyclass(name = "ToolRegistry")]
#[derive(Clone, Default)]
struct PyToolRegistry {
    inner: ToolRegistry,
}

#[pymethods]
impl PyToolRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: ToolRegistry::new(),
        }
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn describe(&self) -> Vec<PyToolDescription> {
        self.inner
            .describe()
            .into_iter()
            .map(|desc| PyToolDescription { inner: desc })
            .collect()
    }
}

#[pyfunction]
fn basic_toolkit_py() -> PyToolRegistry {
    PyToolRegistry {
        inner: basic_toolkit(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry bindings
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "TelemetryLabels")]
#[derive(Clone, Default)]
struct PyTelemetryLabels {
    inner: TelemetryLabels,
}

#[pymethods]
impl PyTelemetryLabels {
    #[new]
    #[pyo3(signature = (tenant=None, tool=None, workflow=None))]
    fn new(tenant: Option<String>, tool: Option<String>, workflow: Option<String>) -> Self {
        Self {
            inner: TelemetryLabels {
                tenant,
                tool,
                workflow,
            },
        }
    }

    #[getter]
    fn tenant(&self) -> Option<String> {
        self.inner.tenant.clone()
    }

    #[setter]
    fn set_tenant(&mut self, tenant: Option<String>) {
        self.inner.tenant = tenant;
    }

    #[getter]
    fn tool(&self) -> Option<String> {
        self.inner.tool.clone()
    }

    #[setter]
    fn set_tool(&mut self, tool: Option<String>) {
        self.inner.tool = tool;
    }

    #[getter]
    fn workflow(&self) -> Option<String> {
        self.inner.workflow.clone()
    }

    #[setter]
    fn set_workflow(&mut self, workflow: Option<String>) {
        self.inner.workflow = workflow;
    }
}

#[pyfunction]
#[pyo3(signature = (service_name, otlp_endpoint=None))]
fn init_tracing_py(service_name: String, otlp_endpoint: Option<String>) -> PyResult<()> {
    sayr_engine::init_tracing(&service_name, otlp_endpoint.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
fn current_span_attributes_py(labels: PyTelemetryLabels) {
    sayr_engine::current_span_attributes(&labels.inner);
}

#[pyfunction]
fn flush_tracer_py() {
    sayr_engine::flush_tracer();
}

#[pyfunction]
fn span_with_labels_py(_name: String, labels: PyTelemetryLabels) {
    let _span = sayr_engine::span_with_labels("python", &labels.inner);
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmarks
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

// ─────────────────────────────────────────────────────────────────────────────
// Stub bindings for remaining exports
// ─────────────────────────────────────────────────────────────────────────────

stub_pyclass!(PyAgentDirective, "AgentDirective");
stub_pyclass!(PyDeploymentPlan, "DeploymentPlan");
stub_pyclass!(PySayrError, "SayrError");

stub_pyclass!(PyAccessController, "AccessController");
stub_pyclass!(PyAction, "Action");
stub_pyclass!(PyPrincipal, "Principal");
stub_pyclass!(PyPrivacyRule, "PrivacyRule");
#[pyclass(name = "GovernanceRole")]
struct PyGovernanceRole;
#[pymethods]
impl PyGovernanceRole {
    #[new]
    fn new() -> PyResult<Self> {
        Err(not_implemented_error("GovernanceRole"))
    }
}

stub_pyclass!(PyAgentHook, "AgentHook");
stub_pyclass!(PyConfirmationHandler, "ConfirmationHandler");

stub_pyclass!(PyDocument, "Document");
stub_pyclass!(PyDocumentChunker, "DocumentChunker");
stub_pyclass!(PyEmbedder, "Embedder");
stub_pyclass!(PyInMemoryVectorStore, "InMemoryVectorStore");
stub_pyclass!(PyKnowledgeBase, "KnowledgeBase");
stub_pyclass!(PyOpenAiEmbedder, "OpenAiEmbedder");
stub_pyclass!(PyOpenAiEmbeddingClient, "OpenAiEmbeddingClient");
stub_pyclass!(PyPgVectorClient, "PgVectorClient");
stub_pyclass!(PyPgVectorStore, "PgVectorStore");
stub_pyclass!(PyQdrantClient, "QdrantClient");
stub_pyclass!(PyQdrantStore, "QdrantStore");
stub_pyclass!(PyRetrievalConfig, "RetrievalConfig");
stub_pyclass!(PyRetrievalEvaluation, "RetrievalEvaluation");
stub_pyclass!(PyRetrievalOverrides, "RetrievalOverrides");
stub_pyclass!(PyRetriever, "Retriever");
stub_pyclass!(PyScoredDocument, "ScoredDocument");
stub_pyclass!(PySearchParams, "SearchParams");
stub_pyclass!(PySimilarityMetric, "SimilarityMetric");
stub_pyclass!(PySlidingWindowChunker, "SlidingWindowChunker");
stub_pyclass!(PyTransformerClient, "TransformerClient");
stub_pyclass!(PyTransformerEmbedder, "TransformerEmbedder");
stub_pyclass!(PyVectorStore, "VectorStore");
stub_pyclass!(PyWhitespaceEmbedder, "WhitespaceEmbedder");

stub_pyclass!(PyAwsBedrockClient, "AwsBedrockClient");
stub_pyclass!(PyAzureOpenAIClient, "AzureOpenAIClient");
stub_pyclass!(PyCohereClient, "CohereClient");
stub_pyclass!(PyFireworksClient, "FireworksClient");
stub_pyclass!(PyGroqClient, "GroqClient");
#[pyclass(name = "LanguageModel")]
struct PyLanguageModel;
#[pymethods]
impl PyLanguageModel {
    #[new]
    fn new() -> PyResult<Self> {
        Err(not_implemented_error("LanguageModel"))
    }
}

stub_pyclass!(PyMistralClient, "MistralClient");
stub_pyclass!(PyModelCompletion, "ModelCompletion");
stub_pyclass!(PyOllamaClient, "OllamaClient");
stub_pyclass!(PyStubModel, "StubModel");
stub_pyclass!(PyTogetherClient, "TogetherClient");

stub_pyclass!(PyConversationMemory, "ConversationMemory");
stub_pyclass!(PyFullMemoryStrategy, "FullMemoryStrategy");
stub_pyclass!(PyMemoryStrategy, "MemoryStrategy");
stub_pyclass!(
    PyPersistentConversationMemory,
    "PersistentConversationMemory"
);
stub_pyclass!(PySummarizedMemoryStrategy, "SummarizedMemoryStrategy");
stub_pyclass!(PyTokenLimitedMemoryStrategy, "TokenLimitedMemoryStrategy");
stub_pyclass!(PyWindowedMemoryStrategy, "WindowedMemoryStrategy");

stub_pyclass!(PyEvaluationReport, "EvaluationReport");
stub_pyclass!(PyMetricsTracker, "MetricsTracker");

stub_pyclass!(PyAgentRuntime, "AgentRuntime");

stub_pyclass!(PyConversationStore, "ConversationStore");
stub_pyclass!(PyFileConversationStore, "FileConversationStore");
stub_pyclass!(PySqlConversationStore, "SqlConversationStore");

stub_pyclass!(PyTeam, "Team");
stub_pyclass!(PyTeamEvent, "TeamEvent");

stub_pyclass!(PyTelemetryCollector, "TelemetryCollector");
stub_pyclass!(PyTelemetrySink, "TelemetrySink");
stub_pyclass!(PyRetryPolicy, "RetryPolicy");
stub_pyclass!(PyFallbackChain, "FallbackChain");

stub_pyclass!(PyAgentTask, "AgentTask");
stub_pyclass!(PyFunctionTask, "FunctionTask");
stub_pyclass!(PyWorkflow, "Workflow");
stub_pyclass!(PyWorkflowContext, "WorkflowContext");
stub_pyclass!(PyWorkflowNode, "WorkflowNode");
stub_pyclass!(PyWorkflowTask, "WorkflowTask");

// ─────────────────────────────────────────────────────────────────────────────
// Module Definition
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn sayr(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Agent>()?;
    m.add_class::<OpenAIChat>()?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_tokens, m)?)?;

    let config = PyModule::new(py, "config")?;
    config.add_class::<PyServerConfig>()?;
    config.add_class::<PySecurityConfig>()?;
    config.add_class::<PyTelemetryConfig>()?;
    config.add_class::<PyDeploymentConfig>()?;
    config.add_class::<PyProviderConfig>()?;
    config.add_class::<PyModelConfig>()?;
    config.add_class::<PyAppConfig>()?;
    m.add_submodule(&config)?;

    let message = PyModule::new(py, "message")?;
    message.add_class::<PyAttachmentKind>()?;
    message.add_class::<PyRole>()?;
    message.add_class::<PyAttachment>()?;
    message.add_class::<PyToolCall>()?;
    message.add_class::<PyToolResult>()?;
    message.add_class::<PyMessage>()?;
    m.add_submodule(&message)?;

    let tools = PyModule::new(py, "tools")?;
    tools.add_class::<PyTool>()?;
    tools.add_class::<PyToolDescription>()?;
    tools.add_class::<PyToolRegistry>()?;
    tools.add_function(wrap_pyfunction!(basic_toolkit_py, &tools)?)?;
    m.add_submodule(&tools)?;

    let telemetry = PyModule::new(py, "telemetry")?;
    telemetry.add_class::<PyTelemetryLabels>()?;
    telemetry.add_function(wrap_pyfunction!(init_tracing_py, &telemetry)?)?;
    telemetry.add_function(wrap_pyfunction!(current_span_attributes_py, &telemetry)?)?;
    telemetry.add_function(wrap_pyfunction!(flush_tracer_py, &telemetry)?)?;
    telemetry.add_function(wrap_pyfunction!(span_with_labels_py, &telemetry)?)?;
    m.add_submodule(&telemetry)?;

    let governance = PyModule::new(py, "governance")?;
    governance.add_class::<PyAccessController>()?;
    governance.add_class::<PyAction>()?;
    governance.add_class::<PyPrincipal>()?;
    governance.add_class::<PyPrivacyRule>()?;
    governance.add_class::<PyGovernanceRole>()?;
    m.add_submodule(&governance)?;

    let knowledge = PyModule::new(py, "knowledge")?;
    knowledge.add_class::<PyDocument>()?;
    knowledge.add_class::<PyDocumentChunker>()?;
    knowledge.add_class::<PyEmbedder>()?;
    knowledge.add_class::<PyInMemoryVectorStore>()?;
    knowledge.add_class::<PyKnowledgeBase>()?;
    knowledge.add_class::<PyOpenAiEmbedder>()?;
    knowledge.add_class::<PyOpenAiEmbeddingClient>()?;
    knowledge.add_class::<PyPgVectorClient>()?;
    knowledge.add_class::<PyPgVectorStore>()?;
    knowledge.add_class::<PyQdrantClient>()?;
    knowledge.add_class::<PyQdrantStore>()?;
    knowledge.add_class::<PyRetrievalConfig>()?;
    knowledge.add_class::<PyRetrievalEvaluation>()?;
    knowledge.add_class::<PyRetrievalOverrides>()?;
    knowledge.add_class::<PyRetriever>()?;
    knowledge.add_class::<PyScoredDocument>()?;
    knowledge.add_class::<PySearchParams>()?;
    knowledge.add_class::<PySimilarityMetric>()?;
    knowledge.add_class::<PySlidingWindowChunker>()?;
    knowledge.add_class::<PyTransformerClient>()?;
    knowledge.add_class::<PyTransformerEmbedder>()?;
    knowledge.add_class::<PyVectorStore>()?;
    knowledge.add_class::<PyWhitespaceEmbedder>()?;
    m.add_submodule(&knowledge)?;

    let llm = PyModule::new(py, "llm")?;
    llm.add_class::<PyAwsBedrockClient>()?;
    llm.add_class::<PyAzureOpenAIClient>()?;
    llm.add_class::<PyCohereClient>()?;
    llm.add_class::<PyFireworksClient>()?;
    llm.add_class::<PyGroqClient>()?;
    llm.add_class::<PyLanguageModel>()?;
    llm.add_class::<PyMistralClient>()?;
    llm.add_class::<PyModelCompletion>()?;
    llm.add_class::<PyOllamaClient>()?;
    llm.add_class::<PyStubModel>()?;
    llm.add_class::<PyTogetherClient>()?;
    m.add_submodule(&llm)?;

    let memory = PyModule::new(py, "memory")?;
    memory.add_class::<PyConversationMemory>()?;
    memory.add_class::<PyFullMemoryStrategy>()?;
    memory.add_class::<PyMemoryStrategy>()?;
    memory.add_class::<PyPersistentConversationMemory>()?;
    memory.add_class::<PySummarizedMemoryStrategy>()?;
    memory.add_class::<PyTokenLimitedMemoryStrategy>()?;
    memory.add_class::<PyWindowedMemoryStrategy>()?;
    m.add_submodule(&memory)?;

    let metrics = PyModule::new(py, "metrics")?;
    metrics.add_class::<PyEvaluationReport>()?;
    metrics.add_class::<PyMetricsTracker>()?;
    m.add_submodule(&metrics)?;

    let server = PyModule::new(py, "server")?;
    server.add_class::<PyAgentRuntime>()?;
    m.add_submodule(&server)?;

    let storage = PyModule::new(py, "storage")?;
    storage.add_class::<PyConversationStore>()?;
    storage.add_class::<PyFileConversationStore>()?;
    storage.add_class::<PySqlConversationStore>()?;
    m.add_submodule(&storage)?;

    let team = PyModule::new(py, "team")?;
    team.add_class::<PyTeam>()?;
    team.add_class::<PyTeamEvent>()?;
    m.add_submodule(&team)?;

    let workflow = PyModule::new(py, "workflow")?;
    workflow.add_class::<PyAgentTask>()?;
    workflow.add_class::<PyFunctionTask>()?;
    workflow.add_class::<PyWorkflow>()?;
    workflow.add_class::<PyWorkflowContext>()?;
    workflow.add_class::<PyWorkflowNode>()?;
    workflow.add_class::<PyWorkflowTask>()?;
    m.add_submodule(&workflow)?;

    let core = PyModule::new(py, "core")?;
    core.add_class::<PyAgentDirective>()?;
    core.add_class::<PyDeploymentPlan>()?;
    core.add_class::<PySayrError>()?;
    // User asked to rename Agno -> Sayr except attribution. Code constructs are not attribution.
    // I already renamed it to PySayrError in my head-patch.
    // Wait, the line above `core.add_class::<PyAgnoError>()?;` needs to match the struct name.
    // I renamed the struct stub to `PySayrError` in the text above. Re-verifying.
    // Yes: stub_pyclass!(PySayrError...
    // So I must register PySayrError.
    core.add_class::<PySayrError>()?; // Corrected to match struct
    core.add_class::<PyAgentHook>()?;
    core.add_class::<PyConfirmationHandler>()?;
    m.add_submodule(&core)?;

    Ok(())
}
