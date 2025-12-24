use serde::{Deserialize, Serialize};

use crate::config::AppConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub name: String,
    pub config: AppConfig,
}

impl DeploymentPlan {
    pub fn render_compose(&self) -> String {
        format!(
            "version: '3'\nservices:\n  agno:\n    image: {}\n    ports:\n      - \"{}:{}\"\n    environment:\n      - AGNO_HOST={}\n      - AGNO_PORT={}\n      - AGNO_TELEMETRY_SAMPLE={}\n",
            self.config
                .deployment
                .container_image
                .clone()
                .unwrap_or_else(|| "ghcr.io/YASSERRMD/sayr-engine:latest".into()),
            self.config.server.port,
            self.config.server.port,
            self.config.server.host,
            self.config.server.port,
            self.config.telemetry.sample_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_compose() {
        let plan = DeploymentPlan {
            name: "demo".into(),
            config: AppConfig::default(),
        };
        let rendered = plan.render_compose();
        assert!(rendered.contains("services:"));
        assert!(rendered.contains("agno"));
    }
}
