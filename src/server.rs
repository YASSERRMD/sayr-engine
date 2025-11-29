use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use tokio::sync::{broadcast, RwLock};
use tokio_stream::wrappers::BroadcastStream;

use crate::llm::LanguageModel;
use crate::{Result, Team, Workflow};

pub struct AgentRuntime<M: LanguageModel + 'static> {
    pub teams: Arc<RwLock<HashMap<String, Team<M>>>>,
    pub workflows: Arc<RwLock<HashMap<String, Arc<Workflow>>>>,
    pub events: broadcast::Sender<String>,
}

impl<M: LanguageModel + 'static> Clone for AgentRuntime<M> {
    fn clone(&self) -> Self {
        Self {
            teams: Arc::clone(&self.teams),
            workflows: Arc::clone(&self.workflows),
            events: self.events.clone(),
        }
    }
}

impl<M: LanguageModel + 'static> AgentRuntime<M> {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(512);
        Self {
            teams: Arc::new(RwLock::new(HashMap::new())),
            workflows: Arc::new(RwLock::new(HashMap::new())),
            events: tx,
        }
    }

    pub async fn register_team(&self, name: impl Into<String>, team: Team<M>) {
        self.teams.write().await.insert(name.into(), team);
    }

    pub async fn register_workflow(&self, flow: Workflow) {
        self.workflows
            .write()
            .await
            .insert(flow.name.clone(), Arc::new(flow));
    }

    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/dashboard", get(dashboard))
            .route("/teams", get(list_teams::<M>))
            .route("/workflows", get(list_workflows::<M>))
            .route("/events", get(stream_events::<M>))
            .route("/invoke", post(run_workflow::<M>))
            .with_state(self.clone());

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app.into_make_service())
            .await
            .map_err(|err| crate::error::AgnoError::Protocol(format!("server error: {err}")))?;
        Ok(())
    }
}

#[derive(Serialize)]
struct TeamSummary {
    name: String,
    agents: usize,
}

async fn list_teams<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
) -> impl IntoResponse {
    let teams = state.teams.read().await;
    let payload: Vec<TeamSummary> = teams
        .iter()
        .map(|(name, team)| TeamSummary {
            name: name.clone(),
            agents: team.size(),
        })
        .collect();
    Json(payload)
}

#[derive(Serialize)]
struct WorkflowSummary {
    name: String,
}

async fn list_workflows<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
) -> impl IntoResponse {
    let flows = state.workflows.read().await;
    let payload: Vec<WorkflowSummary> = flows
        .iter()
        .map(|(name, _)| WorkflowSummary { name: name.clone() })
        .collect();
    Json(payload)
}

async fn stream_events<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let rx = state.events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|msg| async move {
        match msg {
            Ok(line) => Some(Ok::<Event, Infallible>(Event::default().data(line))),
            Err(_) => None,
        }
    });
    Sse::new(stream)
}

#[derive(serde::Deserialize)]
struct WorkflowRequest {
    name: String,
}

async fn run_workflow<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
    Json(req): Json<WorkflowRequest>,
) -> Response {
    let flow = { state.workflows.read().await.get(&req.name).cloned() };
    if let Some(flow) = flow {
        let mut ctx = crate::WorkflowContext::default();
        match flow.run(&mut ctx).await {
            Ok(value) => {
                let _ = state
                    .events
                    .send(format!("workflow:{} completed", flow.name));
                Json(json!({ "result": value, "state": ctx.state })).into_response()
            }
            Err(err) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": err.to_string()})),
            )
                .into_response(),
        }
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({"error":"workflow not found"})),
        )
            .into_response()
    }
}

async fn dashboard() -> Html<&'static str> {
    Html(
        r#"
<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <title>AgentOS Dashboard</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; }
        .column { float: left; width: 45%; margin-right: 5%; }
        .panel { border: 1px solid #ccc; padding: 1rem; margin-bottom: 1rem; border-radius: 8px; }
        h2 { margin-top: 0; }
        #events { background: #111; color: #0f0; height: 200px; overflow: auto; font-family: monospace; padding: 1rem; }
    </style>
</head>
<body>
    <h1>AgentOS</h1>
    <div class="column">
        <div class="panel">
            <h2>Teams</h2>
            <ul id="teams"></ul>
        </div>
        <div class="panel">
            <h2>Workflows</h2>
            <ul id="workflows"></ul>
        </div>
    </div>
    <div class="column">
        <div class="panel">
            <h2>Events</h2>
            <div id="events"></div>
        </div>
    </div>
    <script>
        async function load() {
            const teams = await fetch('/teams').then(r => r.json());
            document.getElementById('teams').innerHTML = teams.map(t => `<li>${t.name} (${t.agents} agents)</li>`).join('');
            const workflows = await fetch('/workflows').then(r => r.json());
            document.getElementById('workflows').innerHTML = workflows.map(w => `<li>${w.name}</li>`).join('');
        }
        load();
        const evt = new EventSource('/events');
        evt.onmessage = (ev) => {
            const node = document.getElementById('events');
            node.innerText += ev.data + "\n";
            node.scrollTop = node.scrollHeight;
        };
    </script>
</body>
</html>
"#,
    )
}
