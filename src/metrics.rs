//! Metrics tracking and evaluation.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram, Meter};
use opentelemetry_prometheus::exporter;
use prometheus::Registry as PromRegistry;
use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::telemetry::TelemetryLabels;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationReport {
    pub duration: Duration,
    pub peak_memory_bytes: u64,
    pub tool_calls: usize,
    pub failures: usize,
    pub success: bool,
    pub labels: TelemetryLabels,
}

impl EvaluationReport {
    pub fn success_rate(reports: &[Self]) -> f32 {
        if reports.is_empty() {
            return 0.0;
        }
        let successes = reports.iter().filter(|r| r.success).count();
        successes as f32 / reports.len() as f32
    }
}

#[derive(Clone)]
pub struct MetricsTracker {
    reports: Arc<Mutex<Vec<EvaluationReport>>>,
    meter: Meter,
    run_counter: Counter<u64>,
    tool_call_counter: Counter<u64>,
    failure_counter: Counter<u64>,
    duration_histogram: Histogram<f64>,
}

impl Default for MetricsTracker {
    fn default() -> Self {
        let meter = global::meter("agno-metrics");
        let run_counter = meter
            .u64_counter("run_total")
            .with_description("Total runs")
            .init();
        let tool_call_counter = meter
            .u64_counter("tool_call_total")
            .with_description("Tool calls")
            .init();
        let failure_counter = meter
            .u64_counter("failure_total")
            .with_description("Failures")
            .init();
        let duration_histogram = meter
            .f64_histogram("run_duration_ms")
            .with_description("Run durations in milliseconds")
            .init();
        Self {
            reports: Arc::new(Mutex::new(Vec::new())),
            meter,
            run_counter,
            tool_call_counter,
            failure_counter,
            duration_histogram,
        }
    }
}

impl MetricsTracker {
    pub fn start_run(&self, labels: TelemetryLabels) -> RunGuard {
        self.run_counter.add(1, &labels.as_attributes());
        RunGuard {
            start: Instant::now(),
            tool_calls: 0,
            failures: 0,
            metrics: self.clone(),
            system: System::new_all(),
            labels,
        }
    }

    pub fn reports(&self) -> Vec<EvaluationReport> {
        self.reports.lock().unwrap().clone()
    }
}

pub struct RunGuard {
    start: Instant,
    tool_calls: usize,
    failures: usize,
    metrics: MetricsTracker,
    system: System,
    labels: TelemetryLabels,
}

impl RunGuard {
    pub fn record_tool_call(&mut self, tool: impl Into<String>) {
        self.tool_calls += 1;
        let labels = self.labels.clone().with_tool(tool.into());
        self.metrics
            .tool_call_counter
            .add(1, &labels.as_attributes());
    }

    pub fn record_failure(&mut self, tool: Option<String>) {
        self.failures += 1;
        let labels = match tool {
            Some(name) => self.labels.clone().with_tool(name),
            None => self.labels.clone(),
        };
        self.metrics.failure_counter.add(1, &labels.as_attributes());
    }

    pub fn finish(mut self, success: bool) -> EvaluationReport {
        let duration = self.start.elapsed();
        self.system.refresh_memory();
        let peak_memory_bytes = self.system.used_memory() * 1024;
        self.metrics
            .duration_histogram
            .record(duration.as_millis() as f64, &self.labels.as_attributes());
        let report = EvaluationReport {
            duration,
            peak_memory_bytes,
            tool_calls: self.tool_calls,
            failures: self.failures,
            success,
            labels: self.labels.clone(),
        };
        self.metrics.reports.lock().unwrap().push(report.clone());
        report
    }
}

pub fn init_prometheus_registry() -> PromRegistry {
    let registry = PromRegistry::new();
    let _ = exporter().with_registry(registry.clone()).build();
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_success_rate_and_labels() {
        let tracker = MetricsTracker::default();
        let labels = TelemetryLabels {
            tenant: Some("tenant-a".into()),
            tool: Some("metrics".into()),
            workflow: Some("test".into()),
        };
        let report = tracker.start_run(labels.clone()).finish(true);
        assert!(report.duration >= Duration::from_millis(0));
        assert_eq!(report.labels, labels);
        let reports = tracker.reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(EvaluationReport::success_rate(&reports), 1.0);
    }
}
