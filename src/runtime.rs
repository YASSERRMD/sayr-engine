use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;

use crate::error::Result;
use crate::llm::LanguageModel;
use crate::team::Team;
use crate::workflow::{Workflow, WorkflowContext, WorkflowEngine};

#[derive(Clone)]
pub struct AgentOs<M: LanguageModel> {
    engine: Arc<WorkflowEngine<M>>,
    workflows: Arc<RwLock<HashMap<String, Workflow<M>>>>,
    events: broadcast::Sender<String>,
}

#[derive(Serialize)]
pub struct TeamResponse {
    name: String,
    members: Vec<String>,
    memory_len: usize,
    state: serde_json::Value,
}

#[derive(Deserialize)]
pub struct RunWorkflowRequest {
    name: String,
}

#[derive(Serialize)]
pub struct RunWorkflowResponse {
    name: String,
    context: WorkflowContext,
}

impl<M: LanguageModel + 'static> AgentOs<M> {
    pub fn new(mut engine: WorkflowEngine<M>) -> Self {
        let (events, _) = broadcast::channel(128);
        engine = engine.with_event_sender(events.clone());
        Self {
            engine: Arc::new(engine),
            workflows: Arc::new(RwLock::new(HashMap::new())),
            events,
        }
    }

    pub async fn register_workflow(&self, workflow: Workflow<M>) {
        let mut locked = self.workflows.write().await;
        locked.insert(workflow.name.clone(), workflow);
    }

    pub fn router(self) -> Router {
        let app_state = Arc::new(self);
        Router::new()
            .route("/", get(Self::dashboard))
            .route("/health", get(Self::health))
            .route("/teams", get(Self::teams))
            .route("/workflows", get(Self::workflows_handler))
            .route("/workflows/run", post(Self::run_workflow))
            .route("/events", get(Self::events))
            .with_state(app_state)
    }

    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        let app = self.router();
        axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
        Ok(())
    }

    async fn dashboard() -> impl IntoResponse {
        Html(include_str!("ui/dashboard.html"))
    }

    async fn health() -> impl IntoResponse {
        "ok"
    }

    async fn teams(State(state): State<Arc<Self>>) -> impl IntoResponse {
        let team = state.engine.team().read().await;
        let shared = team.context().read().await;
        Json(vec![TeamResponse {
            name: team.name().to_string(),
            members: team.member_names(),
            memory_len: shared.memory.len(),
            state: shared.state.clone(),
        }])
    }

    async fn workflows_handler(State(state): State<Arc<Self>>) -> impl IntoResponse {
        let locked = state.workflows.read().await;
        let names: Vec<String> = locked.keys().cloned().collect();
        Json(names)
    }

    async fn run_workflow(
        State(state): State<Arc<Self>>,
        Json(payload): Json<RunWorkflowRequest>,
    ) -> impl IntoResponse {
        let wf = {
            let locked = state.workflows.read().await;
            locked.get(&payload.name).cloned()
        };
        match wf {
            Some(wf) => {
                let _ = state
                    .events
                    .send(format!("workflow:{}:start", payload.name));
                match state.engine.run(&wf).await {
                    Ok(ctx) => {
                        let _ = state
                            .events
                            .send(format!("workflow:{}:complete", payload.name));
                        Json(RunWorkflowResponse {
                            name: payload.name,
                            context: ctx,
                        })
                    }
                    Err(err) => {
                        let _ = state
                            .events
                            .send(format!("workflow:{}:error:{err}", payload.name));
                        (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            err.to_string(),
                        )
                            .into_response()
                    }
                }
            }
            None => (
                axum::http::StatusCode::NOT_FOUND,
                format!("Unknown workflow `{}`", payload.name),
            )
                .into_response(),
        }
    }

    async fn events(
        State(state): State<Arc<Self>>,
    ) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
        let stream = BroadcastStream::new(state.events.subscribe()).map(|msg| {
            msg.map(|m| Event::default().data(m))
                .map_err(|err| axum::Error::new(err.into()))
        });
        Sse::new(stream).keep_alive(KeepAlive::default())
    }
}

/// A helper for quickly spinning up a demo AgentOS with in-memory data.
pub fn demo_agent_os<M: LanguageModel + 'static>(team: Team<M>) -> AgentOs<M> {
    let mut engine = WorkflowEngine::new(team);
    engine.register_task(
        "touch_checkpoint",
        Arc::new(crate::workflow::AsyncFnTask::new(|ctx| {
            Box::pin(async move {
                ctx.write().await.data.insert(
                    "checkpoint".into(),
                    serde_json::Value::String("reached".into()),
                );
                Ok(())
            })
        })),
    );

    AgentOs::new(engine)
}
