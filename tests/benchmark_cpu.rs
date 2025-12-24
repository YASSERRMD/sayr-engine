use std::time::Instant;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn calculate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

#[test]
fn benchmark_cpu() {
    let iterations = 10;
    println!("Starting Pure Rust CPU Benchmark (Iterations: {})", iterations);

    // 1. Fibonacci
    println!("\n--- Benchmarking CPU: Recursive Fibonacci(30) ---");
    let start = Instant::now();
    for _ in 0..iterations {
        fibonacci(30);
    }
    let duration = start.elapsed();
    let avg = duration.as_secs_f64() / iterations as f64;
    println!("3. Agno-Rust (Pure Rust):     {:.6}s", avg);

    // 2. Token Count
    println!("\n--- Benchmarking CPU: Token Count (10MB String) ---");
    let large_text = "word ".repeat(2_000_000);
    let start = Instant::now();
    for _ in 0..iterations {
        calculate_tokens(&large_text);
    }
    let duration = start.elapsed();
    let avg = duration.as_secs_f64() / iterations as f64;
    println!("3. Agno-Rust (Pure Rust):     {:.6}s", avg);
}
