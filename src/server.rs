use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use futures::stream::Stream;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{broadcast, Mutex, RwLock};
use tokio_stream::wrappers::BroadcastStream;

use crate::message::Message;
use crate::{
    AccessController, Action, GovernanceRole, LanguageModel, Principal, Result, SecurityConfig,
    Team, TelemetryCollector, Workflow,
};

pub struct AgentRuntime<M: LanguageModel + 'static> {
    pub teams: Arc<RwLock<HashMap<String, Team<M>>>>,
    pub workflows: Arc<RwLock<HashMap<String, Arc<Workflow>>>>,
    pub agents: Arc<RwLock<HashMap<String, Arc<Mutex<crate::Agent<M>>>>>>,
    pub events: broadcast::Sender<String>,
    trace_events: broadcast::Sender<TraceEvent>,
    security: SecurityConfig,
    access_control: AccessController,
    telemetry: TelemetryCollector,
    metrics: crate::MetricsTracker,
}

impl<M: LanguageModel + 'static> Clone for AgentRuntime<M> {
    fn clone(&self) -> Self {
        Self {
            teams: Arc::clone(&self.teams),
            workflows: Arc::clone(&self.workflows),
            agents: Arc::clone(&self.agents),
            events: self.events.clone(),
            trace_events: self.trace_events.clone(),
            security: self.security.clone(),
            access_control: self.access_control.clone(),
            telemetry: self.telemetry.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

impl<M: LanguageModel + 'static> AgentRuntime<M> {
    pub fn new() -> Self {
        Self::with_security(SecurityConfig::default())
    }

    pub fn with_security(security: SecurityConfig) -> Self {
        let (tx, _) = broadcast::channel(512);
        let (trace_tx, _) = broadcast::channel::<TraceEvent>(256);
        let access_control = AccessController::new();
        access_control.allow(GovernanceRole::User, Action::SendMessage);
        access_control.allow(GovernanceRole::Service, Action::SendMessage);
        access_control.allow(GovernanceRole::Admin, Action::SendMessage);
        access_control.allow(GovernanceRole::User, Action::ReadTranscript);
        access_control.allow(GovernanceRole::Service, Action::ReadTranscript);
        Self {
            teams: Arc::new(RwLock::new(HashMap::new())),
            workflows: Arc::new(RwLock::new(HashMap::new())),
            agents: Arc::new(RwLock::new(HashMap::new())),
            events: tx,
            trace_events: trace_tx,
            security,
            access_control,
            telemetry: TelemetryCollector::default(),
            metrics: crate::MetricsTracker::default(),
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

    pub async fn register_agent(&self, name: impl Into<String>, mut agent: crate::Agent<M>) {
        let name = name.into();
        let controller = Arc::new(self.access_control.clone());
        agent.attach_access_control(controller);
        agent.attach_metrics(self.metrics.clone());
        agent.attach_telemetry(self.telemetry.clone());
        for tool in agent.tool_names() {
            self.access_control
                .allow(GovernanceRole::User, Action::CallTool(tool.clone()));
            self.access_control
                .allow(GovernanceRole::Admin, Action::CallTool(tool.clone()));
            self.access_control
                .allow(GovernanceRole::Service, Action::CallTool(tool.clone()));
        }
        self.agents
            .write()
            .await
            .insert(name, Arc::new(Mutex::new(agent)));
    }

    fn resolve_tenant(
        &self,
        headers: &HeaderMap,
        override_tenant: &Option<String>,
    ) -> std::result::Result<Option<String>, Response> {
        let tenant = override_tenant.clone().or_else(|| {
            headers
                .get("x-tenant")
                .and_then(|h| h.to_str().ok().map(|v| v.to_string()))
        });

        if !self.security.allowed_tenants.is_empty() {
            if let Some(ref t) = tenant {
                if !self
                    .security
                    .allowed_tenants
                    .iter()
                    .any(|allowed| allowed == t)
                {
                    return Err(json_error(
                        StatusCode::UNAUTHORIZED,
                        "tenant not allowed for this deployment",
                    ));
                }
            } else {
                return Err(json_error(
                    StatusCode::UNAUTHORIZED,
                    "tenant is required for this deployment",
                ));
            }
        }

        Ok(tenant)
    }

    fn build_principal(
        &self,
        headers: &HeaderMap,
        body: &AgentChatRequest,
    ) -> std::result::Result<Principal, Response> {
        let tenant = self.resolve_tenant(headers, &body.tenant)?;
        let role = parse_role(
            headers
                .get("x-principal-role")
                .and_then(|h| h.to_str().ok())
                .or_else(|| body.role.as_deref()),
        );
        let principal_id = headers
            .get("x-principal-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
            .or_else(|| body.principal_id.clone())
            .unwrap_or_else(|| "anonymous".into());

        Ok(Principal {
            id: principal_id,
            role,
            tenant,
        })
    }

    fn publish_trace(&self, agent: &str, tenant: Option<String>, kind: TraceKind) {
        let _ = self.trace_events.send(TraceEvent {
            agent: agent.to_string(),
            tenant,
            kind,
        });
    }

    fn emit_tool_traces(&self, agent: &str, tenant: Option<String>, messages: &[Message]) {
        for message in messages {
            if let Some(call) = &message.tool_call {
                self.publish_trace(
                    agent,
                    tenant.clone(),
                    TraceKind::ToolCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                );
            }
            if let Some(result) = &message.tool_result {
                self.publish_trace(
                    agent,
                    tenant.clone(),
                    TraceKind::ToolResult {
                        name: result.name.clone(),
                        output: result.output.clone(),
                    },
                );
            }
        }
    }

    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/dashboard", get(dashboard))
            .route("/agents", get(list_agents::<M>))
            .route("/agents/:id/chat", post(chat_with_agent::<M>))
            .route("/agents/:id/traces", get(stream_tool_traces::<M>))
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

#[derive(Serialize)]
struct AgentSummary {
    name: String,
    tools: usize,
}

#[derive(Deserialize)]
struct AgentChatRequest {
    message: String,
    #[serde(default)]
    principal_id: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    tenant: Option<String>,
}

#[derive(Serialize)]
struct AgentChatResponse {
    reply: String,
    transcript: Vec<Message>,
}

#[derive(Deserialize, Default)]
struct TraceAuth {
    tenant: Option<String>,
    principal_id: Option<String>,
    role: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceEvent {
    agent: String,
    tenant: Option<String>,
    kind: TraceKind,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TraceKind {
    Started { message: String },
    ToolCall { name: String, arguments: Value },
    ToolResult { name: String, output: Value },
    Completed { reply: String },
    Failed { error: String },
}

fn json_error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}

fn parse_role(raw: Option<&str>) -> GovernanceRole {
    match raw.unwrap_or("user").to_lowercase().as_str() {
        "admin" => GovernanceRole::Admin,
        "service" => GovernanceRole::Service,
        _ => GovernanceRole::User,
    }
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

async fn list_agents<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
) -> impl IntoResponse {
    let agents = state.agents.read().await;
    let mut payload = Vec::new();
    for (name, agent) in agents.iter() {
        let guard = agent.lock().await;
        payload.push(AgentSummary {
            name: name.clone(),
            tools: guard.tool_names().len(),
        });
    }
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

async fn stream_tool_traces<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
    Path(agent_id): Path<String>,
    Query(auth): Query<TraceAuth>,
    headers: HeaderMap,
) -> Response {
    let tenant = match state.resolve_tenant(&headers, &auth.tenant) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let principal = match state.build_principal(
        &headers,
        &AgentChatRequest {
            message: String::new(),
            principal_id: auth.principal_id.clone(),
            role: auth.role.clone(),
            tenant: tenant.clone(),
        },
    ) {
        Ok(principal) => principal,
        Err(resp) => return resp,
    };
    if !state
        .access_control
        .authorize(&principal, &Action::ReadTranscript)
    {
        return json_error(
            StatusCode::FORBIDDEN,
            "principal not authorized to read traces",
        );
    }

    state.telemetry.record(
        "sse_subscribe",
        json!({"agent": agent_id.clone(), "tenant": principal.tenant, "principal": principal.id}),
    );

    let rx = state.trace_events.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(move |msg| {
        let tenant = tenant.clone();
        let agent_id = agent_id.clone();
        async move {
            match msg {
                Ok(event) => {
                    if event.agent != agent_id {
                        return None;
                    }
                    if let Some(ref t) = tenant {
                        if event.tenant.as_ref() != Some(t) {
                            return None;
                        }
                    }
                    serde_json::to_string(&event)
                        .ok()
                        .map(|payload| Ok::<Event, Infallible>(Event::default().data(payload)))
                }
                Err(_) => None,
            }
        }
    });
    Sse::new(stream).into_response()
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

async fn chat_with_agent<M: LanguageModel + 'static>(
    State(state): State<AgentRuntime<M>>,
    Path(agent_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<AgentChatRequest>,
) -> Response {
    let principal = match state.build_principal(&headers, &req) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if !state
        .access_control
        .authorize(&principal, &Action::SendMessage)
    {
        return json_error(
            StatusCode::FORBIDDEN,
            "principal not authorized to message this agent",
        );
    }

    let agent = { state.agents.read().await.get(&agent_id).cloned() };
    let Some(agent) = agent else {
        return json_error(StatusCode::NOT_FOUND, "agent not registered");
    };

    state.telemetry.record(
        "http_request",
        json!({"path": format!("/agents/{}/chat", agent_id), "tenant": principal.tenant, "principal": principal.id}),
    );

    let mut guard = agent.lock().await;
    guard.set_principal(principal.clone());
    guard.attach_access_control(Arc::new(state.access_control.clone()));
    guard.attach_metrics(state.metrics.clone());
    guard.attach_telemetry(state.telemetry.clone());

    let starting_len = guard.memory().len();
    state.publish_trace(
        &agent_id,
        principal.tenant.clone(),
        TraceKind::Started {
            message: req.message.clone(),
        },
    );

    let result = guard
        .respond_for(principal.clone(), req.message.clone())
        .await;
    let transcript: Vec<Message> = guard.memory().iter().cloned().collect();
    let new_segment: Vec<Message> = guard.memory().iter().skip(starting_len).cloned().collect();
    drop(guard);

    state.emit_tool_traces(&agent_id, principal.tenant.clone(), &new_segment);

    match result {
        Ok(reply) => {
            state.publish_trace(
                &agent_id,
                principal.tenant.clone(),
                TraceKind::Completed {
                    reply: reply.clone(),
                },
            );
            state.telemetry.record(
                "http_response",
                json!({"path": format!("/agents/{}/chat", agent_id), "status": 200, "tenant": principal.tenant}),
            );
            Json(AgentChatResponse { reply, transcript }).into_response()
        }
        Err(err) => {
            state.publish_trace(
                &agent_id,
                principal.tenant.clone(),
                TraceKind::Failed {
                    error: err.to_string(),
                },
            );
            state.telemetry.record(
                "http_response",
                json!({"path": format!("/agents/{}/chat", agent_id), "status": 502, "error": err.to_string()}),
            );
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": err.to_string()})),
            )
                .into_response()
        }
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
        #trace-log { background: #f6f8fa; height: 200px; overflow: auto; font-family: monospace; padding: 0.75rem; }
        #chat-output { background: #f6f8fa; min-height: 80px; padding: 0.75rem; white-space: pre-wrap; }
        label { display: block; margin-top: 0.5rem; font-weight: 600; }
        input, select, textarea { width: 100%; padding: 0.35rem; margin-top: 0.25rem; }
        button { margin-top: 0.5rem; padding: 0.5rem 1rem; }
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
    <div class="column">
        <div class="panel">
            <h2>Agents</h2>
            <select id="agent-select"></select>
            <div id="agent-tools"></div>
        </div>
        <div class="panel">
            <h2>Chat</h2>
            <label for="tenant">Tenant (x-tenant)</label>
            <input id="tenant" placeholder="acme-co" />
            <label for="principal">Principal ID</label>
            <input id="principal" placeholder="user-123" />
            <label for="role">Role</label>
            <select id="role">
                <option value="user">User</option>
                <option value="admin">Admin</option>
                <option value="service">Service</option>
            </select>
            <label for="chat-input">Message</label>
            <textarea id="chat-input" placeholder="Ask an agent..."></textarea>
            <button onclick="sendChat()">Send</button>
            <div id="chat-output"></div>
        </div>
        <div class="panel">
            <h2>Tool traces (SSE)</h2>
            <div id="trace-log"></div>
        </div>
    </div>
    <script>
        let traceSource = null;
        async function load() {
            const teams = await fetch('/teams').then(r => r.json());
            document.getElementById('teams').innerHTML = teams.map(t => `<li>${t.name} (${t.agents} agents)</li>`).join('');
            const workflows = await fetch('/workflows').then(r => r.json());
            document.getElementById('workflows').innerHTML = workflows.map(w => `<li>${w.name}</li>`).join('');
            await refreshAgents();
        }
        load();
        const evt = new EventSource('/events');
        evt.onmessage = (ev) => {
            const node = document.getElementById('events');
            node.innerText += ev.data + "\n";
            node.scrollTop = node.scrollHeight;
        };

        async function refreshAgents() {
            const agents = await fetch('/agents').then(r => r.json());
            const select = document.getElementById('agent-select');
            select.innerHTML = agents.map(a => `<option value="${a.name}">${a.name} (${a.tools} tools)</option>`).join('');
            if (agents.length) {
                document.getElementById('agent-tools').innerText = `Tools: ${agents[0].tools}`;
                select.value = agents[0].name;
                subscribeTraces();
            } else {
                document.getElementById('agent-tools').innerText = 'No agents registered.';
            }
            select.onchange = () => {
                const selected = agents.find(a => a.name === select.value);
                document.getElementById('agent-tools').innerText = selected ? `Tools: ${selected.tools}` : '';
                subscribeTraces();
            };
        }

        function subscribeTraces() {
            const agent = document.getElementById('agent-select').value;
            if (!agent) return;
            if (traceSource) traceSource.close();
            const params = new URLSearchParams();
            const tenant = document.getElementById('tenant').value;
            const principal = document.getElementById('principal').value;
            const role = document.getElementById('role').value;
            if (tenant) params.append('tenant', tenant);
            if (principal) params.append('principal_id', principal);
            if (role) params.append('role', role);
            const url = `/agents/${agent}/traces${params.toString() ? '?' + params.toString() : ''}`;
            traceSource = new EventSource(url);
            traceSource.onmessage = (ev) => {
                const log = document.getElementById('trace-log');
                try {
                    const data = JSON.parse(ev.data);
                    log.innerText += `[${data.kind}] ${JSON.stringify(data)}\n`;
                } catch (e) {
                    log.innerText += ev.data + "\n";
                }
                log.scrollTop = log.scrollHeight;
            };
        }

        async function sendChat() {
            const agent = document.getElementById('agent-select').value;
            const message = document.getElementById('chat-input').value;
            const tenant = document.getElementById('tenant').value;
            const principal = document.getElementById('principal').value;
            const role = document.getElementById('role').value;
            if (!agent || !message) {
                alert('Select an agent and enter a message.');
                return;
            }
            const headers = {'Content-Type': 'application/json'};
            if (tenant) headers['X-Tenant'] = tenant;
            if (principal) headers['X-Principal-Id'] = principal;
            if (role) headers['X-Principal-Role'] = role;
            const payload = {message, tenant, principal_id: principal, role};
            const res = await fetch(`/agents/${agent}/chat`, {
                method: 'POST',
                headers,
                body: JSON.stringify(payload),
            });
            const body = await res.json();
            if (res.ok) {
                document.getElementById('chat-output').innerText = `Reply: ${body.reply}`;
                subscribeTraces();
            } else {
                document.getElementById('chat-output').innerText = `Error: ${body.error}`;
            }
        }
    </script>
</body>
</html>
"#,
    )
}
