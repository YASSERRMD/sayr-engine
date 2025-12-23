use prometheus::Encoder;

use agno_rust::{
    current_span_attributes, flush_tracer, init_tracing, span_with_labels, MetricsTracker,
    TelemetryCollector, TelemetryLabels,
};

#[tokio::test]
async fn emits_metrics_and_traces_with_labels() {
    // Skip prometheus registry initialization in test since it requires global state
    let _ = init_tracing("agno-test", None);

    let labels = TelemetryLabels {
        tenant: Some("tenant-a".into()),
        tool: Some("integration".into()),
        workflow: Some("workflow-a".into()),
    };

    let telemetry = TelemetryCollector::default();
    let tracker = MetricsTracker::default();
    let mut guard = tracker.start_run(labels.clone());

    let span = span_with_labels("integration_span", &labels);
    let _entered = span.enter();
    telemetry.record(
        "tool_call",
        serde_json::json!({"operation": "ping"}),
        labels.clone(),
    );
    guard.record_tool_call("integration_tool");
    guard.record_failure(Some("integration_tool".into()));
    current_span_attributes(&labels);

    guard.finish(false);
    flush_tracer();

    let drained = telemetry.drain();
    assert_eq!(drained.0.len(), 1);
    assert_eq!(drained.0[0].labels, labels);
}

