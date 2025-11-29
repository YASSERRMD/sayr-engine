use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use tokio::sync::{broadcast, RwLock};

use crate::error::{AgnoError, Result};
use crate::llm::LanguageModel;
use crate::team::Team;

/// Shared scratchpad passed across workflow steps.
#[derive(Default, Debug, Clone)]
pub struct WorkflowContext {
    pub data: HashMap<String, serde_json::Value>,
    pub history: Vec<String>,
}

impl WorkflowContext {
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.data.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }

    pub fn push_history(&mut self, line: impl Into<String>) {
        self.history.push(line.into());
    }
}

pub type WorkflowContextHandle = Arc<RwLock<WorkflowContext>>;

#[async_trait]
pub trait WorkflowTask: Send + Sync {
    async fn run(&self, ctx: WorkflowContextHandle) -> Result<()>;
}

/// Adapter to allow async functions/closures to be used as workflow tasks.
pub struct AsyncFnTask<F>
where
    F: Fn(WorkflowContextHandle) -> futures::future::BoxFuture<'static, Result<()>>
        + Send
        + Sync
        + 'static,
{
    f: Arc<F>,
}

impl<F> AsyncFnTask<F>
where
    F: Fn(WorkflowContextHandle) -> futures::future::BoxFuture<'static, Result<()>>
        + Send
        + Sync
        + 'static,
{
    pub fn new(f: F) -> Self {
        Self { f: Arc::new(f) }
    }
}

#[async_trait]
impl<F> WorkflowTask for AsyncFnTask<F>
where
    F: Fn(WorkflowContextHandle) -> futures::future::BoxFuture<'static, Result<()>>
        + Send
        + Sync
        + 'static,
{
    async fn run(&self, ctx: WorkflowContextHandle) -> Result<()> {
        (self.f)(ctx).await
    }
}

/// A single node in a workflow graph.
pub enum WorkflowStep<M: LanguageModel> {
    Task(String),
    Agent {
        member: String,
        input: String,
    },
    /// Set a value inside the shared workflow context.
    Set {
        key: String,
        value: serde_json::Value,
    },
    Sequence(Vec<WorkflowStep<M>>),
    Parallel(Vec<WorkflowStep<M>>),
    Conditional {
        key: String,
        equals: serde_json::Value,
        then_branch: Box<WorkflowStep<M>>,
        else_branch: Option<Box<WorkflowStep<M>>>,
    },
    Loop {
        key: String,
        until: serde_json::Value,
        body: Box<WorkflowStep<M>>,
        max_iterations: usize,
    },
}

/// A reusable workflow definition.
pub struct Workflow<M: LanguageModel> {
    pub name: String,
    pub steps: Vec<WorkflowStep<M>>,
}

/// Executes workflows that weave teams, tools, and functions together.
pub struct WorkflowEngine<M: LanguageModel> {
    tasks: HashMap<String, Arc<dyn WorkflowTask>>, // plain functions
    team: Arc<RwLock<Team<M>>>,
    events: Option<broadcast::Sender<String>>,
}

impl<M: LanguageModel> WorkflowEngine<M> {
    pub fn new(team: Team<M>) -> Self {
        Self {
            tasks: HashMap::new(),
            team: Arc::new(RwLock::new(team)),
            events: None,
        }
    }

    pub fn register_task(&mut self, name: impl Into<String>, task: Arc<dyn WorkflowTask>) {
        self.tasks.insert(name.into(), task);
    }

    pub fn with_event_sender(mut self, sender: broadcast::Sender<String>) -> Self {
        self.events = Some(sender);
        self
    }

    fn emit(&self, message: impl Into<String>) {
        if let Some(tx) = &self.events {
            let _ = tx.send(message.into());
        }
    }

    pub fn team(&self) -> Arc<RwLock<Team<M>>> {
        Arc::clone(&self.team)
    }

    pub async fn run(&self, workflow: &Workflow<M>) -> Result<WorkflowContext> {
        let ctx: WorkflowContextHandle = Arc::new(RwLock::new(WorkflowContext::default()));
        for step in &workflow.steps {
            self.execute_step(step, Arc::clone(&ctx)).await?;
        }
        let final_ctx = ctx.read().await;
        Ok(final_ctx.clone())
    }

    async fn execute_step(&self, step: &WorkflowStep<M>, ctx: WorkflowContextHandle) -> Result<()> {
        match step {
            WorkflowStep::Task(name) => {
                let Some(task) = self.tasks.get(name) else {
                    return Err(AgnoError::Protocol(format!(
                        "Unknown workflow task `{name}`"
                    )));
                };
                task.run(ctx).await
            }
            WorkflowStep::Agent { member, input } => {
                self.emit(format!("agent:{member}:start"));
                let mut team = self.team.write().await;
                let rendered_input = {
                    let state = ctx.read().await;
                    render_template(input, &state)
                };
                let reply = team.run_agent(member, rendered_input).await?;
                {
                    let mut state = ctx.write().await;
                    state.push_history(format!("{member} -> {reply}"));
                    state.set(
                        format!("reply:{member}"),
                        serde_json::Value::String(reply.clone()),
                    );
                }
                self.emit(format!("agent:{member}:complete"));
                Ok(())
            }
            WorkflowStep::Set { key, value } => {
                let mut state = ctx.write().await;
                state.set(key.clone(), value.clone());
                self.emit(format!("ctx:set:{key}"));
                Ok(())
            }
            WorkflowStep::Sequence(steps) => {
                for sub in steps {
                    self.execute_step(sub, Arc::clone(&ctx)).await?;
                }
                Ok(())
            }
            WorkflowStep::Parallel(steps) => {
                let mut tasks = Vec::new();
                for sub in steps {
                    tasks.push(self.execute_step(sub, Arc::clone(&ctx)));
                }
                let results = join_all(tasks).await;
                for res in results {
                    res?;
                }
                Ok(())
            }
            WorkflowStep::Conditional {
                key,
                equals,
                then_branch,
                else_branch,
            } => {
                let value = ctx
                    .read()
                    .await
                    .data
                    .get(key)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if &value == equals {
                    self.execute_step(then_branch, ctx).await
                } else if let Some(else_branch) = else_branch {
                    self.execute_step(else_branch, ctx).await
                } else {
                    Ok(())
                }
            }
            WorkflowStep::Loop {
                key,
                until,
                body,
                max_iterations,
            } => {
                let mut iterations = 0;
                loop {
                    iterations += 1;
                    if iterations > *max_iterations {
                        return Err(AgnoError::Protocol(format!(
                            "Loop on `{key}` exceeded {max_iterations} iterations"
                        )));
                    }
                    self.execute_step(body, Arc::clone(&ctx)).await?;
                    let value = ctx
                        .read()
                        .await
                        .data
                        .get(key)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    if &value == until {
                        break;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Render a simple `{{key}}` template string using values from the workflow context.
fn render_template(template: &str, ctx: &WorkflowContext) -> String {
    let mut rendered = String::new();
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' && chars.peek() == Some(&'{') {
            // consume the second '{'
            let _ = chars.next();
            let mut key = String::new();
            while let Some(next) = chars.next() {
                if next == '}' && chars.peek() == Some(&'}') {
                    let _ = chars.next();
                    break;
                }
                key.push(next);
            }
            let replacement = ctx
                .get(key.trim())
                .and_then(|v| {
                    if v.is_string() {
                        v.as_str().map(|s| s.to_string())
                    } else {
                        Some(v.to_string())
                    }
                })
                .unwrap_or_default();
            rendered.push_str(&replacement);
        } else {
            rendered.push(ch);
        }
    }

    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn sets_and_reads_context_values() {
        let mut team = Team::new("demo");
        let model =
            crate::llm::StubModel::new(vec![r#"{"action":"respond","content":"ack"}"#.to_string()]);
        team.add_member("alice", Agent::new(model));

        let engine = WorkflowEngine::new(team);
        let workflow = Workflow {
            name: "ctx".into(),
            steps: vec![
                WorkflowStep::Set {
                    key: "topic".into(),
                    value: json!("rust"),
                },
                WorkflowStep::Agent {
                    member: "alice".into(),
                    input: "teach me about {{topic}}".into(),
                },
            ],
        };

        let ctx = engine.run(&workflow).await.unwrap();
        assert_eq!(ctx.data.get("topic"), Some(&json!("rust")));
        assert_eq!(ctx.data.get("reply:alice"), Some(&json!("ack")));
        assert_eq!(ctx.history, vec!["alice -> ack".to_string()]);
    }

    #[tokio::test]
    async fn executes_tasks_and_conditionals() {
        let mut team = Team::new("demo");
        let model =
            crate::llm::StubModel::new(
                vec![r#"{"action":"respond","content":"done"}"#.to_string()],
            );
        team.add_member("worker", Agent::new(model));

        let mut engine = WorkflowEngine::new(team);
        engine.register_task(
            "seed",
            Arc::new(AsyncFnTask::new(|ctx| {
                Box::pin(async move {
                    ctx.write().await.set("ready", json!(true));
                    Ok(())
                })
            })),
        );

        let workflow = Workflow {
            name: "conditional".into(),
            steps: vec![
                WorkflowStep::Task("seed".into()),
                WorkflowStep::Conditional {
                    key: "ready".into(),
                    equals: json!(true),
                    then_branch: Box::new(WorkflowStep::Agent {
                        member: "worker".into(),
                        input: "start".into(),
                    }),
                    else_branch: Some(Box::new(WorkflowStep::Set {
                        key: "skipped".into(),
                        value: json!(true),
                    })),
                },
            ],
        };

        let ctx = engine.run(&workflow).await.unwrap();
        assert_eq!(ctx.data.get("ready"), Some(&json!(true)));
        assert_eq!(ctx.data.get("reply:worker"), Some(&json!("done")));
        assert_eq!(ctx.history.len(), 1);
    }
}
