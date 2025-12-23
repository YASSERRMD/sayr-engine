use opentelemetry::global;
use opentelemetry::sdk::trace::TracerProvider;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use prometheus::Encoder;

use agno_rust::metrics::{init_prometheus_registry, MetricsTracker};
use agno_rust::telemetry::{
    current_span_attributes, flush_tracer, init_tracing, span_with_labels, TelemetryCollector,
    TelemetryLabels,
};

#[tokio::test]
async fn emits_metrics_and_traces_with_labels() {
    let registry = init_prometheus_registry();
    init_tracing("agno-test", None).expect("tracing should initialize");

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

    let metric_families = registry.gather();
    let mut buffer = Vec::new();
    prometheus::TextEncoder::new()
        .encode(&metric_families, &mut buffer)
        .expect("encode metrics");
    let encoded = String::from_utf8(buffer).expect("utf8");
    assert!(encoded.contains("run_total"));
    assert!(encoded.contains("tool_call_total"));

    // Ensure tracer provider attached
    let provider = global::trace_provider();
    assert!(provider.downcast_ref::<TracerProvider>().is_some());
}
