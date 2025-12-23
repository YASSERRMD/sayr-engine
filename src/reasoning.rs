//! Reasoning module for chain-of-thought agent orchestration.
//!
//! Provides structured reasoning with step-by-step analysis, validation,
//! and confidence scoring.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::Result;
use crate::llm::LanguageModel;
use crate::message::Message;

/// A single reasoning step in the chain-of-thought process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step number
    pub step: usize,
    /// Title of this step
    pub title: String,
    /// Action being taken
    pub action: String,
    /// Result of the action
    pub result: Option<String>,
    /// Reasoning behind this step
    pub reasoning: String,
    /// Next action to take
    pub next_action: NextAction,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// Next action to take after a reasoning step
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NextAction {
    Continue,
    Validate,
    FinalAnswer,
    Reset,
}

/// Collection of reasoning steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSteps {
    pub steps: Vec<ReasoningStep>,
    pub final_answer: Option<String>,
}

impl ReasoningSteps {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_answer: None,
        }
    }

    pub fn add_step(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }

    pub fn set_final_answer(&mut self, answer: String) {
        self.final_answer = Some(answer);
    }

    pub fn total_confidence(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.steps.iter().map(|s| s.confidence).sum::<f32>() / self.steps.len() as f32
    }
}

impl Default for ReasoningSteps {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the reasoning agent
#[derive(Clone)]
pub struct ReasoningConfig {
    /// Minimum number of reasoning steps
    pub min_steps: usize,
    /// Maximum number of reasoning steps
    pub max_steps: usize,
    /// Whether to use structured JSON output
    pub use_json_mode: bool,
    /// Enable debug logging
    pub debug_mode: bool,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            min_steps: 3,
            max_steps: 10,
            use_json_mode: true,
            debug_mode: false,
        }
    }
}

/// Generate the reasoning agent system prompt
pub fn reasoning_system_prompt(config: &ReasoningConfig) -> String {
    format!(
        r#"You are a meticulous, thoughtful, and logical Reasoning Agent who solves complex problems through clear, structured, step-by-step analysis.

## Process

### Step 1 - Problem Analysis:
- Restate the user's task clearly in your own words
- Identify what information is required and what tools might be necessary

### Step 2 - Decompose and Strategize:
- Break down the problem into clearly defined subtasks
- Develop at least two distinct strategies or approaches

### Step 3 - Intent Clarification and Planning:
- Articulate the user's intent behind their request
- Select the most suitable strategy and justify your choice
- Formulate a detailed step-by-step action plan

### Step 4 - Execute the Action Plan:
For each step, document:
1. **Title**: Concise summary of the step
2. **Action**: State your next action ('I will...')
3. **Result**: Execute and summarize the outcome
4. **Reasoning**: Explain your rationale (necessity, considerations, progression, assumptions)
5. **Next Action**: Choose continue/validate/final_answer/reset
6. **Confidence Score**: 0.0-1.0 indicating certainty

### Step 5 - Validation:
- Cross-verify with alternative approaches
- Document validation results
- If validation fails, reset and revise

### Step 6 - Final Answer:
- Deliver your solution clearly and succinctly
- Restate how it addresses the original task

## Guidelines:
- Ensure analysis is complete, comprehensive, logical, and actionable
- Handle errors by resetting or revising immediately
- Use minimum {} and maximum {} steps
- Execute tools proactively when needed"#,
        config.min_steps, config.max_steps
    )
}

/// A reasoning agent that uses chain-of-thought to solve problems
pub struct ReasoningAgent<M: LanguageModel> {
    model: Arc<M>,
    config: ReasoningConfig,
}

impl<M: LanguageModel> ReasoningAgent<M> {
    pub fn new(model: Arc<M>, config: ReasoningConfig) -> Self {
        Self { model, config }
    }

    /// Run the reasoning process on a given problem
    pub async fn reason(&self, problem: &str) -> Result<ReasoningSteps> {
        let system_prompt = reasoning_system_prompt(&self.config);

        let messages = vec![
            Message::system(&system_prompt),
            Message::user(problem),
        ];

        // Get initial response from model
        let completion = self.model.complete_chat(&messages, &[], false).await?;

        // Try to parse as structured reasoning
        if let Some(content) = completion.content {
            // Try to extract JSON from the response
            if let Some(json_start) = content.find('{') {
                if let Some(json_end) = content.rfind('}') {
                    let json_str = &content[json_start..=json_end];
                    if let Ok(steps) = serde_json::from_str::<ReasoningSteps>(json_str) {
                        return Ok(steps);
                    }
                }
            }

            // Fallback: create a simple reasoning step from the response
            let mut steps = ReasoningSteps::new();
            steps.add_step(ReasoningStep {
                step: 1,
                title: "Analysis".into(),
                action: "Analyzed the problem".into(),
                result: Some(content.clone()),
                reasoning: "Direct analysis of the problem".into(),
                next_action: NextAction::FinalAnswer,
                confidence: 0.8,
            });
            steps.set_final_answer(content);
            return Ok(steps);
        }

        // Empty response
        Ok(ReasoningSteps::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_steps() {
        let mut steps = ReasoningSteps::new();
        steps.add_step(ReasoningStep {
            step: 1,
            title: "Test".into(),
            action: "Testing".into(),
            result: Some("Success".into()),
            reasoning: "For testing".into(),
            next_action: NextAction::Continue,
            confidence: 0.9,
        });
        steps.add_step(ReasoningStep {
            step: 2,
            title: "Final".into(),
            action: "Finalizing".into(),
            result: Some("Done".into()),
            reasoning: "Completing".into(),
            next_action: NextAction::FinalAnswer,
            confidence: 0.95,
        });

        assert_eq!(steps.steps.len(), 2);
        assert!((steps.total_confidence() - 0.925).abs() < 0.001);
    }

    #[test]
    fn test_reasoning_prompt_generation() {
        let config = ReasoningConfig::default();
        let prompt = reasoning_system_prompt(&config);
        assert!(prompt.contains("Reasoning Agent"));
        assert!(prompt.contains("step-by-step"));
    }
}
