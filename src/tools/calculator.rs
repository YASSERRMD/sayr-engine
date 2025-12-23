//! Calculator toolkit.
//!
//! Provides basic math operations: add, subtract, multiply, divide,
//! exponentiate, factorial, is_prime, and square_root.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{AgnoError, Result};
use crate::tool::{Tool, ToolRegistry};

/// Create a Calculator toolkit with all math operations
pub fn calculator_toolkit() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(AddTool);
    registry.register(SubtractTool);
    registry.register(MultiplyTool);
    registry.register(DivideTool);
    registry.register(ExponentiateTool);
    registry.register(FactorialTool);
    registry.register(IsPrimeTool);
    registry.register(SquareRootTool);
    registry
}

struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "add"
    }

    fn description(&self) -> &str {
        "Add two numbers. Expects {\"a\": number, \"b\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let a = get_number(&input, "a", "add")?;
        let b = get_number(&input, "b", "add")?;
        let result = a + b;
        Ok(json!({ "operation": "addition", "result": result }))
    }
}

struct SubtractTool;

#[async_trait]
impl Tool for SubtractTool {
    fn name(&self) -> &str {
        "subtract"
    }

    fn description(&self) -> &str {
        "Subtract second number from first. Expects {\"a\": number, \"b\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let a = get_number(&input, "a", "subtract")?;
        let b = get_number(&input, "b", "subtract")?;
        let result = a - b;
        Ok(json!({ "operation": "subtraction", "result": result }))
    }
}

struct MultiplyTool;

#[async_trait]
impl Tool for MultiplyTool {
    fn name(&self) -> &str {
        "multiply"
    }

    fn description(&self) -> &str {
        "Multiply two numbers. Expects {\"a\": number, \"b\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let a = get_number(&input, "a", "multiply")?;
        let b = get_number(&input, "b", "multiply")?;
        let result = a * b;
        Ok(json!({ "operation": "multiplication", "result": result }))
    }
}

struct DivideTool;

#[async_trait]
impl Tool for DivideTool {
    fn name(&self) -> &str {
        "divide"
    }

    fn description(&self) -> &str {
        "Divide first number by second. Expects {\"a\": number, \"b\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let a = get_number(&input, "a", "divide")?;
        let b = get_number(&input, "b", "divide")?;
        if b == 0.0 {
            return Ok(json!({ "operation": "division", "error": "Division by zero is undefined" }));
        }
        let result = a / b;
        Ok(json!({ "operation": "division", "result": result }))
    }
}

struct ExponentiateTool;

#[async_trait]
impl Tool for ExponentiateTool {
    fn name(&self) -> &str {
        "exponentiate"
    }

    fn description(&self) -> &str {
        "Raise first number to the power of second. Expects {\"a\": number, \"b\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let a = get_number(&input, "a", "exponentiate")?;
        let b = get_number(&input, "b", "exponentiate")?;
        let result = a.powf(b);
        Ok(json!({ "operation": "exponentiation", "result": result }))
    }
}

struct FactorialTool;

#[async_trait]
impl Tool for FactorialTool {
    fn name(&self) -> &str {
        "factorial"
    }

    fn description(&self) -> &str {
        "Calculate factorial of a non-negative integer. Expects {\"n\": integer}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let n = input
            .get("n")
            .and_then(Value::as_i64)
            .ok_or_else(|| AgnoError::Protocol("missing `n` for factorial".into()))?;

        if n < 0 {
            return Ok(
                json!({ "operation": "factorial", "error": "Factorial of a negative number is undefined" }),
            );
        }

        let result = factorial(n as u64);
        Ok(json!({ "operation": "factorial", "result": result }))
    }
}

struct IsPrimeTool;

#[async_trait]
impl Tool for IsPrimeTool {
    fn name(&self) -> &str {
        "is_prime"
    }

    fn description(&self) -> &str {
        "Check if a number is prime. Expects {\"n\": integer}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let n = input
            .get("n")
            .and_then(Value::as_i64)
            .ok_or_else(|| AgnoError::Protocol("missing `n` for is_prime".into()))?;

        let result = is_prime(n);
        Ok(json!({ "operation": "prime_check", "result": result }))
    }
}

struct SquareRootTool;

#[async_trait]
impl Tool for SquareRootTool {
    fn name(&self) -> &str {
        "square_root"
    }

    fn description(&self) -> &str {
        "Calculate square root of a non-negative number. Expects {\"n\": number}."
    }

    async fn call(&self, input: Value) -> Result<Value> {
        let n = get_number(&input, "n", "square_root")?;
        if n < 0.0 {
            return Ok(
                json!({ "operation": "square_root", "error": "Square root of a negative number is undefined" }),
            );
        }
        let result = n.sqrt();
        Ok(json!({ "operation": "square_root", "result": result }))
    }
}

// Helper functions

fn get_number(input: &Value, field: &str, tool_name: &str) -> Result<f64> {
    input
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| AgnoError::Protocol(format!("missing `{}` for {}", field, tool_name)))
}

fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

fn is_prime(n: i64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add() {
        let registry = calculator_toolkit();
        let add = registry.get("add").unwrap();
        let result = add.call(json!({"a": 2, "b": 3})).await.unwrap();
        assert_eq!(result["result"], 5.0);
    }

    #[tokio::test]
    async fn test_factorial() {
        let registry = calculator_toolkit();
        let factorial = registry.get("factorial").unwrap();
        let result = factorial.call(json!({"n": 5})).await.unwrap();
        assert_eq!(result["result"], 120);
    }

    #[tokio::test]
    async fn test_is_prime() {
        let registry = calculator_toolkit();
        let is_prime = registry.get("is_prime").unwrap();

        let result = is_prime.call(json!({"n": 7})).await.unwrap();
        assert_eq!(result["result"], true);

        let result = is_prime.call(json!({"n": 4})).await.unwrap();
        assert_eq!(result["result"], false);
    }
}
