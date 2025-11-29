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
            },
        }
    }
}

impl AppConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&raw).map_err(|err| {
            AgnoError::Protocol(format!("Failed to parse configuration: {err}"))
        })?;
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
        if let Ok(sample) = env::var("AGNO_TELEMETRY_SAMPLE") {
            if let Ok(parsed) = sample.parse::<f32>() {
                cfg.telemetry.sample_rate = parsed.clamp(0.01, 1.0);
            }
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
}
