//! Integration test for telemetry and metrics functionality.

use sayr_engine::{MetricsTracker, TelemetryCollector, TelemetryLabels};

#[tokio::test]
async fn emits_metrics_and_traces_with_labels() {
    let labels = TelemetryLabels {
        tenant: Some("tenant-a".into()),
        tool: Some("integration".into()),
        workflow: Some("workflow-a".into()),
    };

    let telemetry = TelemetryCollector::default();
    let tracker = MetricsTracker::default();
    let mut guard = tracker.start_run(labels.clone());

    telemetry.record(
        "tool_call",
        serde_json::json!({"operation": "ping"}),
        labels.clone(),
    );
    guard.record_tool_call("integration_tool");
    guard.record_failure(Some("integration_tool".into()));

    let report = guard.finish(false);
    assert!(!report.success);
    assert_eq!(report.tool_calls, 1);
    assert_eq!(report.failures, 1);
    assert_eq!(report.labels, labels);

    let drained = telemetry.drain();
    assert_eq!(drained.0.len(), 1);
    assert_eq!(drained.0[0].labels, labels);
}
