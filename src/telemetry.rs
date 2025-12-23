use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use opentelemetry::global;
use opentelemetry::trace::{Span, SpanKind, Tracer};
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk;
use serde::{Deserialize, Serialize};
use serde_json;
use tokio::time::sleep;
use tracing::{span, Level};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

use crate::error::{AgnoError, Result};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TelemetryLabels {
    pub tenant: Option<String>,
    pub tool: Option<String>,
    pub workflow: Option<String>,
}

impl TelemetryLabels {
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    pub fn with_workflow(mut self, workflow: impl Into<String>) -> Self {
        self.workflow = Some(workflow.into());
        self
    }

    pub fn as_attributes(&self) -> Vec<KeyValue> {
        let mut attrs = Vec::new();
        if let Some(tenant) = &self.tenant {
            attrs.push(KeyValue::new("tenant", tenant.clone()));
        }
        if let Some(tool) = &self.tool {
            attrs.push(KeyValue::new("tool", tool.clone()));
        }
        if let Some(workflow) = &self.workflow {
            attrs.push(KeyValue::new("workflow", workflow.clone()));
        }
        attrs
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub kind: String,
    pub timestamp: SystemTime,
    pub detail: serde_json::Value,
    pub labels: TelemetryLabels,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub context: String,
    pub error: String,
    pub attempt: u32,
    pub labels: TelemetryLabels,
}

#[derive(Default, Clone)]
pub struct TelemetryCollector {
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
    failures: Arc<Mutex<Vec<FailureRecord>>>,
}

impl TelemetryCollector {
    pub fn record(
        &self,
        kind: impl Into<String>,
        detail: serde_json::Value,
        labels: TelemetryLabels,
    ) {
        self.events.lock().unwrap().push(TelemetryEvent {
            kind: kind.into(),
            timestamp: SystemTime::now(),
            detail,
            labels,
        });
    }

    pub fn record_failure(
        &self,
        context: impl Into<String>,
        error: impl Into<String>,
        attempt: u32,
        labels: TelemetryLabels,
    ) {
        self.failures.lock().unwrap().push(FailureRecord {
            context: context.into(),
            error: error.into(),
            attempt,
            labels,
        });
    }

    pub fn drain(&self) -> (Vec<TelemetryEvent>, Vec<FailureRecord>) {
        let mut events = self.events.lock().unwrap();
        let mut failures = self.failures.lock().unwrap();
        (std::mem::take(&mut *events), std::mem::take(&mut *failures))
    }
}

#[derive(Default, Clone)]
pub struct TelemetrySink {
    buffer: Arc<Mutex<Vec<TelemetryEvent>>>,
}

impl TelemetrySink {
    pub fn push(&self, event: TelemetryEvent) {
        self.buffer.lock().unwrap().push(event);
    }

    pub fn flush(&self) -> Vec<TelemetryEvent> {
        let mut guard = self.buffer.lock().unwrap();
        std::mem::take(&mut *guard)
    }
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff: Duration,
}

impl RetryPolicy {
    pub fn default_external_call() -> Self {
        Self {
            max_retries: 3,
            backoff: Duration::from_millis(200),
        }
    }

    pub async fn retry<F, Fut, T>(
        &self,
        mut f: F,
        telemetry: Option<&TelemetryCollector>,
        labels: TelemetryLabels,
    ) -> Result<T>
    where
        F: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        for attempt in 0..=self.max_retries {
            match f(attempt).await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if let Some(t) = telemetry {
                        t.record_failure("retry", format!("{err}"), attempt, labels.clone());
                    }
                    let span = span!(
                        Level::INFO,
                        "retry_failure",
                        attempt,
                        tenant = labels.tenant.as_deref().unwrap_or(""),
                        tool = labels.tool.as_deref().unwrap_or(""),
                        workflow = labels.workflow.as_deref().unwrap_or("")
                    );
                    let _enter = span.enter();
                    tracing::warn!("retry attempt {} failed: {}", attempt, err);
                    if attempt == self.max_retries {
                        return Err(err);
                    }
                    sleep(self.backoff * (attempt + 1)).await;
                }
            }
        }
        Err(AgnoError::Protocol("retry exhausted".into()))
    }
}

#[derive(Clone)]
pub struct FallbackChain<T> {
    steps: Vec<(String, Arc<dyn Fn() -> Result<T> + Send + Sync>)>,
}

impl<T> std::fmt::Debug for FallbackChain<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let labels: Vec<&str> = self.steps.iter().map(|(label, _)| label.as_str()).collect();
        f.debug_struct("FallbackChain")
            .field("steps", &labels)
            .finish()
    }
}

impl<T> FallbackChain<T> {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn with_step(
        mut self,
        label: impl Into<String>,
        handler: impl Fn() -> Result<T> + Send + Sync + 'static,
    ) -> Self {
        self.steps.push((label.into(), Arc::new(handler)));
        self
    }

    pub fn execute(
        &self,
        telemetry: Option<&TelemetryCollector>,
        labels: TelemetryLabels,
    ) -> Result<T> {
        let mut last_error: Option<AgnoError> = None;
        for (label, handler) in self.steps.iter() {
            let span = span!(
                Level::DEBUG,
                "fallback_step",
                step = label.as_str(),
                tenant = labels.tenant.as_deref().unwrap_or(""),
                tool = labels.tool.as_deref().unwrap_or(""),
                workflow = labels.workflow.as_deref().unwrap_or("")
            );
            let _guard = span.enter();
            match handler() {
                Ok(value) => {
                    if let Some(t) = telemetry {
                        t.record(
                            "fallback_success",
                            serde_json::json!({ "step": label }),
                            labels.clone(),
                        );
                    }
                    tracing::info!("fallback step succeeded");
                    return Ok(value);
                }
                Err(err) => {
                    if let Some(t) = telemetry {
                        t.record_failure(label.clone(), format!("{err}"), 0, labels.clone());
                    }
                    tracing::warn!("fallback step failed: {}", err);
                    last_error = Some(err);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| AgnoError::Protocol("fallback exhausted".into())))
    }
}

pub fn span_with_labels(_name: &str, labels: &TelemetryLabels) -> tracing::Span {
    span!(
        Level::INFO,
        "labeled_span",
        tenant = labels.tenant.as_deref().unwrap_or(""),
        tool = labels.tool.as_deref().unwrap_or(""),
        workflow = labels.workflow.as_deref().unwrap_or("")
    )
}

pub fn init_tracing(service_name: &str, otlp_endpoint: Option<&str>) -> Result<()> {
    let trace_config = opentelemetry_sdk::trace::config().with_resource(
        opentelemetry_sdk::Resource::new(vec![KeyValue::new(
            "service.name",
            service_name.to_owned(),
        )]),
    );

    let tracer = if let Some(endpoint) = otlp_endpoint {
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_trace_config(trace_config)
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .map_err(|e| AgnoError::Telemetry(e.to_string()))?
    } else {
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_trace_config(trace_config)
            .with_exporter(opentelemetry_otlp::new_exporter().tonic())
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .map_err(|e| AgnoError::Telemetry(e.to_string()))?
    };

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    let fmt_layer = tracing_subscriber::fmt::layer().json().with_target(true);
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    Registry::default()
        .with(env_filter)
        .with(fmt_layer)
        .with(telemetry)
        .try_init()
        .map_err(|e| AgnoError::Telemetry(format!("failed to init tracing: {e}")))?;
    Ok(())
}

pub fn current_span_attributes(labels: &TelemetryLabels) {
    let tracer = global::tracer("agno-tracer");
    let mut span = tracer
        .span_builder("context")
        .with_kind(SpanKind::Internal)
        .with_attributes(labels.as_attributes())
        .start(&tracer);
    span.add_event("context attached".to_string(), labels.as_attributes());
    span.end();
}

pub fn flush_tracer() {
    // OpenTelemetry 0.22 doesn't expose a flush method; shutdown flushes internally
    global::shutdown_tracer_provider();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn retries_until_success() {
        let policy = RetryPolicy {
            max_retries: 2,
            backoff: Duration::from_millis(1),
        };
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let calls = Arc::new(Mutex::new(0u32));
        let telemetry = TelemetryCollector::default();
        let labels = TelemetryLabels {
            tenant: Some("tenant-a".into()),
            tool: Some("retry".into()),
            workflow: Some("test".into()),
        };
        let res = policy
            .retry(
                |_: u32| {
                    let calls = calls.clone();
                    async move {
                        let mut guard = calls.lock().await;
                        *guard += 1;
                        if *guard < 2 {
                            Err(AgnoError::Protocol("fail".into()))
                        } else {
                            Ok(42)
                        }
                    }
                },
                Some(&telemetry),
                labels.clone(),
            )
            .await;
        assert_eq!(res.unwrap(), 42);
        let drained = telemetry.drain();
        assert_eq!(drained.1.len(), 1);
        assert_eq!(drained.1[0].labels, labels);
    }

    #[test]
    fn runs_fallbacks() {
        let telemetry = TelemetryCollector::default();
        let labels = TelemetryLabels {
            tenant: Some("tenant-a".into()),
            tool: Some("fallback".into()),
            workflow: Some("test".into()),
        };
        let chain = FallbackChain::new()
            .with_step("primary", || Err(AgnoError::Protocol("nope".into())))
            .with_step("secondary", || Ok("ok"));
        let res = chain.execute(Some(&telemetry), labels.clone()).unwrap();
        assert_eq!(res, "ok");
        let drained = telemetry.drain();
        assert_eq!(drained.1.len(), 1);
        assert_eq!(drained.1[0].labels, labels);
    }
}
