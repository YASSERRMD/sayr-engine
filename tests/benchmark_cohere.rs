use sayr_engine::{CohereClient, LanguageModel, Message, Role};
use std::time::Instant;
use std::env;

#[tokio::test]
#[ignore]
async fn benchmark_cohere() {
    // Load env manually or assume set
    let api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let model_id = "command-a-03-2025";
    let iterations = 10;

    let client = CohereClient::new(api_key).with_model(model_id);

    println!("Starting Rust Benchmark (Model: {}, Iterations: {})", model_id, iterations);
    println!("--- Testing Pure Rust (agno-rust CohereClient) ---");

    let mut latencies = Vec::new();

    for i in range(0, iterations) {
        print!("Iteration {}/{}...", i + 1, iterations);
        let start = Instant::now();

        let messages = vec![Message {
            role: Role::User,
            content: "Say hello in one word.".to_string(),
            tool_call: None,
            tool_result: None,
            attachments: vec![],
        }];

        match client.complete_chat(&messages, &[], false).await {
            Ok(_) => {
                let duration = start.elapsed();
                latencies.push(duration);
                println!(" {:.4}s", duration.as_secs_f64());
            }
            Err(e) => {
                println!(" FAILED: {}", e);
            }
        }
    }

    if !latencies.is_empty() {
        let sum: f64 = latencies.iter().map(|d| d.as_secs_f64()).sum();
        let avg = sum / latencies.len() as f64;
        println!("Pure Rust Average: {:.4}s", avg);
    }
}

fn range(start: i32, end: i32) -> impl Iterator<Item = i32> {
    start..end
}
