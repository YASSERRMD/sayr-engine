use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationReport {
    pub duration: Duration,
    pub peak_memory_bytes: u64,
    pub tool_calls: usize,
    pub failures: usize,
    pub success: bool,
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

#[derive(Default, Clone)]
pub struct MetricsTracker {
    reports: Arc<Mutex<Vec<EvaluationReport>>>,
}

impl MetricsTracker {
    pub fn start_run(&self) -> RunGuard {
        RunGuard {
            start: Instant::now(),
            tool_calls: 0,
            failures: 0,
            metrics: self.clone(),
            system: System::new_all(),
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
}

impl RunGuard {
    pub fn record_tool_call(&mut self) {
        self.tool_calls += 1;
    }

    pub fn record_failure(&mut self) {
        self.failures += 1;
    }

    pub fn finish(mut self, success: bool) -> EvaluationReport {
        let duration = self.start.elapsed();
        self.system.refresh_memory();
        let peak_memory_bytes = self.system.used_memory() * 1024;
        let report = EvaluationReport {
            duration,
            peak_memory_bytes,
            tool_calls: self.tool_calls,
            failures: self.failures,
            success,
        };
        self.metrics.reports.lock().unwrap().push(report.clone());
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_success_rate() {
        let tracker = MetricsTracker::default();
        let report = tracker.start_run().finish(true);
        assert!(report.duration >= Duration::from_millis(0));
        let reports = tracker.reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(EvaluationReport::success_rate(&reports), 1.0);
    }
}
