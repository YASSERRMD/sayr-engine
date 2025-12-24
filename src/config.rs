use std::env;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AgnoError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_tls")]
    pub tls_enabled: bool,
}

fn default_tls() -> bool {
    false
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityConfig {
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default)]
    pub allowed_tenants: Vec<String>,
    #[serde(default = "default_encryption_required")]
    pub encryption_required: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            allowed_origins: Vec::new(),
            allowed_tenants: Vec::new(),
            encryption_required: default_encryption_required(),
        }
    }
}

fn default_encryption_required() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TelemetryConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: f32,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_retention_hours")]
    pub retention_hours: u32,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            endpoint: None,
            retention_hours: default_retention_hours(),
        }
    }
}

fn default_sample_rate() -> f32 {
    1.0
}

fn default_retention_hours() -> u32 {
    72
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeploymentConfig {
    #[serde(default = "default_replicas")]
    pub replicas: u16,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: u32,
    #[serde(default)]
    pub autoscale: bool,
    #[serde(default)]
    pub container_image: Option<String>,
}

impl Default for DeploymentConfig {
    fn default() -> Self {
        Self {
            replicas: default_replicas(),
            max_concurrency: default_max_concurrency(),
            autoscale: false,
            container_image: None,
        }
    }
}

fn default_replicas() -> u16 {
    1
}

fn default_max_concurrency() -> u32 {
    32
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub openai: ProviderConfig,
    #[serde(default)]
    pub anthropic: ProviderConfig,
    #[serde(default)]
    pub gemini: ProviderConfig,
    #[serde(default)]
    pub cohere: ProviderConfig,
    #[cfg(feature = "aws")]
    #[serde(default)]
    pub bedrock: ProviderConfig,
}

#[cfg(feature = "persistence")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
}
#[cfg(not(feature = "persistence"))]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
}

#[cfg(feature = "persistence")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    File,
    Sqlite,
}

#[cfg(feature = "persistence")]
impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::File
    }
}

#[cfg(feature = "persistence")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StorageConfig {
    #[serde(default)]
    pub backend: StorageBackend,
    #[serde(default = "default_storage_path")]
    pub file_path: String,
    #[serde(default)]
    pub database_url: Option<String>,
}

#[cfg(feature = "persistence")]
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::default(),
            file_path: default_storage_path(),
            database_url: None,
        }
    }
}

#[cfg(feature = "persistence")]
fn default_storage_path() -> String {
    "conversation.jsonl".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    pub server: ServerConfig,
    #[serde(default)]
    pub security: SecurityConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub deployment: DeploymentConfig,
    pub model: ModelConfig,
    #[cfg(feature = "persistence")]
    #[serde(default)]
    pub storage: StorageConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".into(),
                port: 8080,
                tls_enabled: default_tls(),
            },
            security: SecurityConfig {
                allowed_origins: vec![],
                allowed_tenants: vec![],
                encryption_required: default_encryption_required(),
            },
            telemetry: TelemetryConfig {
                sample_rate: default_sample_rate(),
                endpoint: None,
                retention_hours: default_retention_hours(),
            },
            deployment: DeploymentConfig {
                replicas: default_replicas(),
                max_concurrency: default_max_concurrency(),
                autoscale: false,
                container_image: None,
            },
            model: ModelConfig {
                provider: "stub".into(),
                model: "stub-model".into(),
                api_key: None,
                base_url: None,
                organization: None,
                stream: false,
                openai: ProviderConfig::default(),
                anthropic: ProviderConfig::default(),
                gemini: ProviderConfig::default(),
                cohere: ProviderConfig::default(),
                #[cfg(feature = "aws")]
                bedrock: ProviderConfig::default(),
            },
            #[cfg(feature = "persistence")]
            storage: StorageConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&raw)
            .map_err(|err| AgnoError::Protocol(format!("Failed to parse configuration: {err}")))?;
        Ok(cfg)
    }

    pub fn from_env_or_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut cfg = Self::from_file(path)?;
        if let Ok(host) = env::var("AGNO_HOST") {
            cfg.server.host = host;
        }
        if let Ok(port) = env::var("AGNO_PORT") {
            if let Ok(parsed) = port.parse::<u16>() {
                cfg.server.port = parsed;
            }
        }
        if let Ok(key) = env::var("AGNO_API_KEY") {
            cfg.model.api_key = Some(key);
        }
        if let Ok(openai_key) = env::var("AGNO_OPENAI_API_KEY") {
            cfg.model.openai.api_key = Some(openai_key);
        }
        if let Ok(openai_endpoint) = env::var("AGNO_OPENAI_ENDPOINT") {
            cfg.model.openai.endpoint = Some(openai_endpoint);
        }
        if let Ok(openai_org) = env::var("AGNO_OPENAI_ORG") {
            cfg.model.openai.organization = Some(openai_org);
        }
        if let Ok(anthropic_key) = env::var("AGNO_ANTHROPIC_API_KEY") {
            cfg.model.anthropic.api_key = Some(anthropic_key);
        }
        if let Ok(anthropic_endpoint) = env::var("AGNO_ANTHROPIC_ENDPOINT") {
            cfg.model.anthropic.endpoint = Some(anthropic_endpoint);
        }
        if let Ok(gemini_key) = env::var("AGNO_GEMINI_API_KEY") {
            cfg.model.gemini.api_key = Some(gemini_key);
        }
        if let Ok(gemini_endpoint) = env::var("AGNO_GEMINI_ENDPOINT") {
            cfg.model.gemini.endpoint = Some(gemini_endpoint);
        }
        if let Ok(cohere_key) = env::var("AGNO_COHERE_API_KEY") {
            cfg.model.cohere.api_key = Some(cohere_key);
        }
        if let Ok(cohere_endpoint) = env::var("AGNO_COHERE_ENDPOINT") {
            cfg.model.cohere.endpoint = Some(cohere_endpoint);
        }
        if let Ok(stream) = env::var("AGNO_STREAMING") {
            if let Ok(parsed) = stream.parse::<bool>() {
                cfg.model.stream = parsed;
            }
        }
        if let Ok(sample) = env::var("AGNO_TELEMETRY_SAMPLE") {
            if let Ok(parsed) = sample.parse::<f32>() {
                cfg.telemetry.sample_rate = parsed.clamp(0.01, 1.0);
            }
        }
        if let Ok(_backend) = env::var("AGNO_STORAGE_BACKEND") {
            #[cfg(feature = "persistence")]
            {
                cfg.storage.backend = match _backend.to_ascii_lowercase().as_str() {
                    "sqlite" => StorageBackend::Sqlite,
                    _ => StorageBackend::File,
                };
            }
        }
        if let Ok(_path) = env::var("AGNO_STORAGE_PATH") {
            #[cfg(feature = "persistence")]
            { cfg.storage.file_path = _path; }
        }
        if let Ok(_url) = env::var("AGNO_DATABASE_URL") {
            #[cfg(feature = "persistence")]
            { cfg.storage.database_url = Some(_url); }
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_and_overrides() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[server]\nhost='127.0.0.1'\nport=9000\n[model]\nprovider='openai'\nmodel='gpt-4'"
        )
        .unwrap();

        env::set_var("AGNO_PORT", "9100");
        let cfg = AppConfig::from_env_or_file(file.path()).unwrap();

        assert_eq!(cfg.server.port, 9100);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.model.provider, "openai");
        env::remove_var("AGNO_PORT");
    }

    #[test]
    #[cfg(feature = "persistence")]
    fn overrides_storage_backend() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[server]\nhost='127.0.0.1'\nport=9000\n[model]\nprovider='openai'\nmodel='gpt-4'\n[storage]\nbackend='file'\nfile_path='transcript.jsonl'"
        )
        .unwrap();

        env::set_var("AGNO_STORAGE_BACKEND", "sqlite");
        env::set_var("AGNO_DATABASE_URL", "sqlite::memory:");
        let cfg = AppConfig::from_env_or_file(file.path()).unwrap();

        assert_eq!(cfg.storage.backend, StorageBackend::Sqlite);
        assert_eq!(
            cfg.storage.database_url,
            Some("sqlite::memory:".to_string())
        );

        env::remove_var("AGNO_STORAGE_BACKEND");
        env::remove_var("AGNO_DATABASE_URL");
    }
}
