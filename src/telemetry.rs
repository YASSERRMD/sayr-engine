use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use crate::error::{AgnoError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub kind: String,
    pub timestamp: SystemTime,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub context: String,
    pub error: String,
    pub attempt: u32,
}

#[derive(Default, Clone)]
pub struct TelemetryCollector {
    events: Arc<Mutex<Vec<TelemetryEvent>>>,
    failures: Arc<Mutex<Vec<FailureRecord>>>,
}

impl TelemetryCollector {
    pub fn record(&self, kind: impl Into<String>, detail: serde_json::Value) {
        self.events.lock().unwrap().push(TelemetryEvent {
            kind: kind.into(),
            timestamp: SystemTime::now(),
            detail,
        });
    }

    pub fn record_failure(
        &self,
        context: impl Into<String>,
        error: impl Into<String>,
        attempt: u32,
    ) {
        self.failures.lock().unwrap().push(FailureRecord {
            context: context.into(),
            error: error.into(),
            attempt,
        });
    }

    pub fn drain(&self) -> (Vec<TelemetryEvent>, Vec<FailureRecord>) {
        let mut events = self.events.lock().unwrap();
        let mut failures = self.failures.lock().unwrap();
        (std::mem::take(&mut *events), std::mem::take(&mut *failures))
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
                        t.record_failure("retry", format!("{err}"), attempt);
                    }
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

    pub fn execute(&self, telemetry: Option<&TelemetryCollector>) -> Result<T> {
        let mut last_error: Option<AgnoError> = None;
        for (label, handler) in self.steps.iter() {
            match handler() {
                Ok(value) => {
                    if let Some(t) = telemetry {
                        t.record("fallback_success", serde_json::json!({"step": label}));
                    }
                    return Ok(value);
                }
                Err(err) => {
                    if let Some(t) = telemetry {
                        t.record_failure(label.clone(), format!("{err}"), 0);
                    }
                    last_error = Some(err);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| AgnoError::Protocol("fallback exhausted".into())))
    }
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
            )
            .await;
        assert_eq!(res.unwrap(), 42);
        assert!(!telemetry.drain().1.is_empty());
    }

    #[test]
    fn runs_fallbacks() {
        let telemetry = TelemetryCollector::default();
        let chain = FallbackChain::new()
            .with_step("primary", || Err(AgnoError::Protocol("nope".into())))
            .with_step("secondary", || Ok("ok"));
        let res = chain.execute(Some(&telemetry)).unwrap();
        assert_eq!(res, "ok");
        let drained = telemetry.drain();
        assert_eq!(drained.1.len(), 1);
    }
}
