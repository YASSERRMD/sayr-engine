use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::agent::Agent;
use crate::{LanguageModel, Result};

/// Shared state threaded through a workflow execution.
#[derive(Debug, Clone, Default)]
pub struct WorkflowContext {
    pub state: Map<String, Value>,
    pub logs: Vec<String>,
}

impl WorkflowContext {
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }
}

#[async_trait]
pub trait WorkflowTask: Send + Sync {
    async fn run(&self, ctx: &mut WorkflowContext) -> Result<Value>;
}

type TaskFuture<'a> = Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;

/// Wrap a plain async function as a workflow task.
pub struct FunctionTask<F>
where
    F: for<'a> Fn(&'a mut WorkflowContext) -> TaskFuture<'a> + Send + Sync,
{
    func: F,
}

impl<F> FunctionTask<F>
where
    F: for<'a> Fn(&'a mut WorkflowContext) -> TaskFuture<'a> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F> WorkflowTask for FunctionTask<F>
where
    F: for<'a> Fn(&'a mut WorkflowContext) -> TaskFuture<'a> + Send + Sync,
{
    async fn run(&self, ctx: &mut WorkflowContext) -> Result<Value> {
        (self.func)(ctx).await
    }
}

/// Task that dispatches to an individual agent and stores the reply under a key.
pub struct AgentTask<M: LanguageModel> {
    agent: Arc<tokio::sync::Mutex<Agent<M>>>,
    prompt_key: Option<String>,
    store_under: Option<String>,
    fallback_prompt: String,
}

impl<M: LanguageModel> AgentTask<M> {
    pub fn new(
        agent: Arc<tokio::sync::Mutex<Agent<M>>>,
        prompt_key: Option<String>,
        store_under: Option<String>,
        fallback_prompt: impl Into<String>,
    ) -> Self {
        Self {
            agent,
            prompt_key,
            store_under,
            fallback_prompt: fallback_prompt.into(),
        }
    }
}

#[async_trait]
impl<M: LanguageModel> WorkflowTask for AgentTask<M> {
    async fn run(&self, ctx: &mut WorkflowContext) -> Result<Value> {
        let prompt = self
            .prompt_key
            .as_ref()
            .and_then(|k| ctx.get(k))
            .and_then(|v| v.as_str())
            .unwrap_or(&self.fallback_prompt)
            .to_string();
        let mut agent = self.agent.lock().await;
        let reply = agent.respond(prompt).await?;
        let value = Value::String(reply.clone());
        if let Some(key) = &self.store_under {
            ctx.insert(key.clone(), value.clone());
        }
        Ok(value)
    }
}

pub type Condition = Arc<dyn Fn(&WorkflowContext) -> bool + Send + Sync>;

#[derive(Clone)]
pub enum WorkflowNode {
    Task(Arc<dyn WorkflowTask>),
    Sequence(Vec<WorkflowNode>),
    Parallel(Vec<WorkflowNode>),
    Conditional {
        condition: Condition,
        then_branch: Box<WorkflowNode>,
        else_branch: Option<Box<WorkflowNode>>,
    },
    Loop {
        condition: Condition,
        body: Box<WorkflowNode>,
        max_iterations: usize,
    },
}

impl WorkflowNode {
    fn execute<'a>(
        &'a self,
        ctx: &'a mut WorkflowContext,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            match self {
                WorkflowNode::Task(task) => task.run(ctx).await,
                WorkflowNode::Sequence(steps) => {
                    let mut last = Value::Null;
                    for step in steps {
                        last = step.execute(ctx).await?;
                    }
                    Ok(last)
                }
                WorkflowNode::Parallel(steps) => {
                    let mut combined = Vec::new();
                    for step in steps {
                        combined.push(step.execute(ctx).await?);
                    }
                    Ok(Value::Array(combined))
                }
                WorkflowNode::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    if condition(ctx) {
                        then_branch.execute(ctx).await
                    } else if let Some(other) = else_branch {
                        other.execute(ctx).await
                    } else {
                        Ok(Value::Null)
                    }
                }
                WorkflowNode::Loop {
                    condition,
                    body,
                    max_iterations,
                } => {
                    let mut last = Value::Null;
                    for _ in 0..*max_iterations {
                        if !(condition)(ctx) {
                            break;
                        }
                        last = body.execute(ctx).await?;
                    }
                    Ok(last)
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct Workflow {
    pub name: String,
    pub root: WorkflowNode,
}

impl Workflow {
    pub fn new(name: impl Into<String>, root: WorkflowNode) -> Self {
        Self {
            name: name.into(),
            root,
        }
    }

    pub async fn run(&self, ctx: &mut WorkflowContext) -> Result<Value> {
        self.root.execute(ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn executes_sequential_and_parallel_nodes() {
        let task_a = FunctionTask::new(|ctx: &mut WorkflowContext| {
            Box::pin(async move {
                ctx.insert("a", json!(1));
                Ok(json!("done"))
            })
        });
        let task_b = FunctionTask::new(|ctx: &mut WorkflowContext| {
            Box::pin(async move {
                let current = ctx.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                ctx.insert("b", json!(current + 1));
                Ok(json!("b"))
            })
        });

        let flow = Workflow::new(
            "demo",
            WorkflowNode::Sequence(vec![
                WorkflowNode::Task(Arc::new(task_a)),
                WorkflowNode::Parallel(vec![
                    WorkflowNode::Task(Arc::new(task_b)),
                    WorkflowNode::Task(Arc::new(FunctionTask::new(|ctx: &mut WorkflowContext| {
                        Box::pin(async move {
                            ctx.insert("c", json!(true));
                            Ok(json!("c"))
                        })
                    }))),
                ]),
            ]),
        );

        let mut ctx = WorkflowContext::default();
        let result = flow.run(&mut ctx).await.unwrap();
        assert!(result.is_array());
        assert_eq!(ctx.get("a").unwrap(), &json!(1));
        assert_eq!(ctx.get("b").unwrap(), &json!(2));
        assert_eq!(ctx.get("c").unwrap(), &json!(true));
    }

    #[tokio::test]
    async fn executes_conditional_loop() {
        let body = WorkflowNode::Task(Arc::new(FunctionTask::new(|ctx: &mut WorkflowContext| {
            Box::pin(async move {
                let next = ctx.get("count").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
                ctx.insert("count", json!(next));
                Ok(json!(next))
            })
        })));

        let condition: Condition = Arc::new(|ctx: &WorkflowContext| {
            ctx.get("count").and_then(|v| v.as_i64()).unwrap_or(0) < 3
        });

        let flow = Workflow::new(
            "looping",
            WorkflowNode::Loop {
                condition,
                body: Box::new(body),
                max_iterations: 10,
            },
        );

        let mut ctx = WorkflowContext::default();
        ctx.insert("count", json!(0));
        flow.run(&mut ctx).await.unwrap();
        assert_eq!(ctx.get("count").unwrap(), &json!(3));
    }
}
