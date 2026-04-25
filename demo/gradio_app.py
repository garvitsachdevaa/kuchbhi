"""
SpindleFlow RL — Professional Gradio Dashboard
================================================
Run:  cd spindleflow-rl && python demo/gradio_app.py
URL:  http://localhost:7860
"""

from __future__ import annotations
import os, sys, json, html, threading
from pathlib import Path
import numpy as np

# Use cached models only — avoids HuggingFace Hub network calls at startup
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env.spindleflow_env import SpindleFlowEnv
from env.state import EpisodeState
from env.specialist_registry import SpecialistRegistry

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

CONFIG  = "configs/training_config.yaml"
CATALOG = "configs/specialist_catalog.yaml"
ASSETS  = Path("demo/assets")

SPEC_COLORS = {
    "frontend_react":      "#00d4ff",
    "backend_api":         "#7c3aed",
    "database_architect":  "#f59e0b",
    "devops_engineer":     "#10b981",
    "security_analyst":    "#ef4444",
    "product_strategist":  "#8b5cf6",
    "ux_designer":         "#ec4899",
    "tech_writer":         "#94a3b8",
}

PRESET_TASKS = [
    "Design a microservices auth system with JWT, OAuth2, and rate limiting",
    "Build a real-time chat app with WebSockets and React",
    "Create a data pipeline processing 1M daily transactions",
    "Design CI/CD for a monorepo with 5 microservices",
    "Write API docs for a REST payment processing service",
    "Design a database schema for an e-commerce platform",
    "Build a secure file upload system with virus scanning",
    "Create a Kubernetes zero-downtime deployment strategy",
]

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif"),
    margin=dict(l=44, r=20, t=44, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
)

# ─────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────

class Session:
    def __init__(self):
        self.env: SpindleFlowEnv | None = None
        self.registry: SpecialistRegistry | None = None
        self.rewards: list[float] = []
        self.actions: list[dict] = []
        self.step_n = 0
        self.done = False
        self.task = ""

    def boot(self):
        if self.env is None:
            self.env = SpindleFlowEnv(
                config_path=CONFIG, catalog_path=CATALOG,
                use_real_spindleflow=False, phase=1,
            )
            self.registry = self.env.registry

    def reset(self, phase: int = 1):
        self.boot()
        self.env.phase = int(phase)
        obs, info = self.env.reset()
        self.rewards, self.actions, self.step_n, self.done = [], [], 0, False
        self.task = info.get("task", "")
        return obs, info

    def step(self, action):
        if self.env is None or self.done:
            return None, 0.0, True, False, {}
        obs, r, term, trunc, info = self.env.step(action)
        self.rewards.append(r)
        self.actions.append(info)
        self.step_n += 1
        self.done = term or trunc
        return obs, r, term, trunc, info

S = Session()
# Pre-warm sentence-transformer on startup so first Reset is instant
_prewarm = threading.Thread(target=S.boot, daemon=True)
_prewarm.start()

# ─────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────

def fig_reward_curve(rewards: list[float]) -> go.Figure:
    if not rewards:
        fig = go.Figure()
        fig.update_layout(
            **DARK,
            title=dict(text="Episode Reward", font=dict(size=13, color="#64748b")),
            annotations=[dict(text="Reset the environment to begin", x=0.5, y=0.5,
                              showarrow=False, font=dict(color="#334155", size=13))],
        )
        return fig

    steps = list(range(len(rewards)))
    cumul = np.cumsum(rewards).tolist()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=steps, y=cumul, mode="lines",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        name="Cumulative",
    ), row=1, col=1)

    bar_colors = ["#10b981" if r >= 0 else "#ef4444" for r in rewards]
    fig.add_trace(go.Bar(
        x=steps, y=rewards, marker_color=bar_colors,
        marker_line_width=0, name="Per-step",
    ), row=2, col=1)

    fig.update_layout(**DARK, height=300, showlegend=False,
                      title=dict(text="Episode Reward", font=dict(size=13, color="#94a3b8")))
    fig.update_yaxes(title_text="Cumul.", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Step", row=2, col=1, title_font_size=10)
    return fig


def fig_delegation_graph(called_ids: list[str], edges: list[tuple]) -> go.Figure:
    nodes = ["orchestrator"] + [c for c in called_ids if c != "orchestrator"]
    all_ids = list(S.registry.list_ids()) if S.registry else []
    # add dimmed uncalled nodes
    uncalled = [x for x in all_ids if x not in nodes]
    full_nodes = nodes + uncalled

    n = len(full_nodes)
    angles = [2 * np.pi * i / max(n, 1) for i in range(n)]
    pos = {nd: (np.cos(a), np.sin(a)) for nd, a in zip(full_nodes, angles)}

    fig = go.Figure()

    # edges
    for src, dst in edges:
        if src in pos and dst in pos:
            x0, y0 = pos[src]; x1, y1 = pos[dst]
            fig.add_trace(go.Scatter(
                x=[x0, (x0+x1)/2, x1, None], y=[y0, (y0+y1)/2, y1, None],
                mode="lines", line=dict(color="rgba(0,212,255,0.45)", width=2),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                ax=x0, ay=y0, x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowsize=1.2, arrowwidth=2,
                arrowcolor="rgba(0,212,255,0.7)", showarrow=True,
            )

    # nodes
    for nd in full_nodes:
        x, y = pos[nd]
        is_orch   = nd == "orchestrator"
        is_called = nd in called_ids
        color   = "#f59e0b" if is_orch else (SPEC_COLORS.get(nd, "#7c3aed") if is_called else "#1e293b")
        size    = 32 if is_orch else (20 if is_called else 13)
        opacity = 1.0 if (is_orch or is_called) else 0.28
        label   = nd.replace("_", "\n")

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, opacity=opacity,
                        line=dict(color="rgba(255,255,255,0.15)", width=1.5)),
            text=[label], textposition="top center",
            textfont=dict(size=8, color=f"rgba(226,232,240,{opacity})"),
            hovertext=[f"<b>{nd}</b>{'  (called)' if is_called else ''}"],
            hoverinfo="text", showlegend=False,
        ))

    _graph_layout = {k: v for k, v in DARK.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **_graph_layout,
        title=dict(text="Delegation Graph", font=dict(size=13, color="#94a3b8")),
        height=340,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
    )
    return fig


def fig_reward_breakdown(components: dict) -> go.Figure:
    if not components:
        components = {k: 0.0 for k in [
            "quality_delta", "efficiency_penalty", "failure_penalty",
            "recovery_bonus", "conflict_penalty", "conflict_bonus",
            "consistency_bonus", "latency_penalty", "explanation_bonus",
        ]}
    names  = list(components.keys())
    values = [components[k] for k in names]
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]
    labels = [n.replace("_", " ").title() for n in names]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside", textfont=dict(color="#94a3b8", size=9),
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(**DARK, height=310,
                      title=dict(text="Reward Breakdown", font=dict(size=13, color="#94a3b8")),
                      xaxis_title="Value")
    return fig


def fig_similarity(registry: SpecialistRegistry) -> go.Figure:
    ids = registry.list_ids()
    n   = len(ids)
    mat = np.zeros((n, n))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            ea = registry.get(a).to_state_vector()
            eb = registry.get(b).to_state_vector()
            mat[i][j] = float(np.dot(ea, eb))

    labels = [x.replace("_", "<br>") for x in ids]
    fig = go.Figure(go.Heatmap(
        z=mat, x=labels, y=labels,
        colorscale=[[0,"#0f0f1a"],[0.5,"rgba(124,58,237,0.6)"],[1,"#00d4ff"]],
        showscale=True, zmin=0, zmax=1,
        text=np.round(mat, 2), texttemplate="%{text}", textfont=dict(size=9),
    ))
    fig.update_layout(**DARK, height=400,
                      title=dict(text="Capability Similarity (Cosine)", font=dict(size=13, color="#94a3b8")))
    return fig


def fig_training_curve() -> go.Figure:
    path = ASSETS / "reward_curve.json"
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        eps, rews = d["episodes"], d["mean_rewards"]
    else:
        eps  = list(range(0, 201, 5))
        rews = [float(np.clip(0.1 + 0.5*(1-np.exp(-e/80)) + np.random.normal(0, 0.04), 0, 1))
                for e in eps]

    smooth = [float(np.mean(rews[max(0,i-4):i+1])) for i in range(len(rews))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=rews, mode="markers",
                             marker=dict(size=5, color="rgba(0,212,255,0.35)"),
                             name="Episode"))
    fig.add_trace(go.Scatter(x=eps, y=smooth, mode="lines",
                             line=dict(color="#00d4ff", width=2.5),
                             fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
                             name="Smoothed"))
    fig.add_hline(y=0.1, line_dash="dash", line_color="rgba(148,163,184,0.35)",
                  annotation_text="Random baseline", annotation_font_color="#64748b")
    fig.update_layout(**DARK, height=340,
                      title=dict(text="Training Progress — Mean Reward", font=dict(size=13, color="#94a3b8")),
                      xaxis_title="Episode", yaxis_title="Mean Reward",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def fig_policy_compare() -> go.Figure:
    path = ASSETS / "demo_moment_2.json"
    if not path.exists():
        return go.Figure()
    with open(path) as f:
        d = json.load(f)
    qp, lp = d["quality_policy"], d["latency_policy"]
    cats = ["Specialists", "Est. Time (s)", "Latency Weight ×100"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Quality Policy",
                         x=cats, y=[len(qp["specialists_called"]), qp["estimated_time_s"], qp["latency_weight"]*100],
                         marker_color="#7c3aed", marker_line_width=0))
    fig.add_trace(go.Bar(name="Latency Policy",
                         x=cats, y=[len(lp["specialists_called"]), lp["estimated_time_s"], lp["latency_weight"]*100],
                         marker_color="#00d4ff", marker_line_width=0))
    fig.update_layout(**DARK, barmode="group", height=320,
                      title=dict(text="Quality vs Latency Policy", font=dict(size=13, color="#94a3b8")),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


# ─────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────

def _hero() -> str:
    return """
<div style="background:linear-gradient(135deg,#0f0f1a,#130a22,#091422);
            border:1px solid rgba(0,212,255,0.14);border-radius:16px;
            padding:28px 36px;margin-bottom:2px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-60px;right:-40px;width:360px;height:360px;
              background:radial-gradient(circle,rgba(124,58,237,0.11) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:15%;width:280px;height:280px;
              background:radial-gradient(circle,rgba(0,212,255,0.07) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="font-size:26px;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#7c3aed,#00d4ff);
              background-size:200% auto;-webkit-background-clip:text;
              -webkit-text-fill-color:transparent;background-clip:text;
              margin:0 0 5px 0;letter-spacing:-0.3px;">SpindleFlow RL</div>
  <div style="color:#64748b;font-size:13px;margin:0 0 18px 0;">
    Delegation Policy Learning Environment &mdash; Teaching orchestrators to route, specialize, and stop.
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;
                 background:rgba(0,212,255,0.1);color:#00d4ff;border:1px solid rgba(0,212,255,0.22);">OPENENV v0</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;
                 background:rgba(124,58,237,0.1);color:#a78bfa;border:1px solid rgba(124,58,237,0.22);">LSTM PPO</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;
                 background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.22);">20/20 TESTS</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;
                 background:rgba(245,158,11,0.1);color:#fbbf24;border:1px solid rgba(245,158,11,0.22);">HACKATHON 2026</span>
    <span style="display:inline-flex;align-items:center;gap:5px;padding:3px 13px;border-radius:999px;
                 font-size:10px;font-weight:700;letter-spacing:0.5px;
                 background:rgba(16,185,129,0.08);color:#34d399;border:1px solid rgba(16,185,129,0.25);">
      <span style="width:6px;height:6px;border-radius:50%;background:#10b981;
                   box-shadow:0 0 6px #10b981;animation:pdot 2s infinite;display:inline-block;"></span>
      OPENENV COMPLIANT
    </span>
  </div>
</div>
<style>
@keyframes pdot{0%,100%{opacity:1;box-shadow:0 0 6px #10b981}50%{opacity:.5;box-shadow:0 0 14px #10b981}}
</style>
"""


def _metrics(obs_dim: int, act_dim: int, n_spec: int, phase: int) -> str:
    items = [
        (str(obs_dim), "Obs Dim", "#00d4ff"),
        (str(act_dim), "Action Dim", "#7c3aed"),
        (str(n_spec), "Specialists", "#10b981"),
        (f"Phase {phase}", "Curriculum", "#f59e0b"),
    ]
    cards = "".join(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:16px 18px;transition:all .2s;">
  <div style="font-size:24px;font-weight:700;color:{c};line-height:1;margin-bottom:4px;">{v}</div>
  <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;">{l}</div>
</div>""" for v, l, c in items)
    return f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 4px;">{cards}</div>'


def _spec_cards(registry: SpecialistRegistry) -> str:
    cards = ""
    for sp in registry.list_all():
        c = SPEC_COLORS.get(sp.id, "#7c3aed")
        cards += f"""
<div style="background:rgba(255,255,255,0.025);border:1px solid {c}18;border-left:3px solid {c};
            border-radius:12px;padding:14px;transition:all .2s;">
  <div style="font-size:11px;font-weight:700;color:{c};margin-bottom:6px;">
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                 background:{c};box-shadow:0 0 6px {c}80;margin-right:5px;"></span>
    {sp.role}
  </div>
  <div style="font-size:11px;color:#64748b;line-height:1.5;">{html.escape(sp.description[:88])}…</div>
  <div style="font-size:10px;color:#334155;margin-top:8px;padding-top:8px;
              border-top:1px solid rgba(255,255,255,0.05);">
    {sp.avg_latency_ms}ms avg &nbsp;·&nbsp; {', '.join(sp.complexity_affinity)}
  </div>
</div>"""
    return f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0;">{cards}</div>'


def _sec(title: str) -> str:
    return f"""<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
    letter-spacing:1px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.07);
    margin-bottom:14px;">{title}</div>"""


def _log_html(actions: list[dict], rewards: list[float]) -> str:
    if not actions:
        body = "  Waiting…  Reset the episode to start."
    else:
        lines = []
        for i, (info, r) in enumerate(zip(actions, rewards)):
            sign = "+" if r >= 0 else ""
            color = "#10b981" if r >= 0 else "#ef4444"
            act = html.escape(info.get("action_name", "UNKNOWN"))
            specs = info.get("called_specialists", [])
            mode  = info.get("delegation_mode", "")
            lines.append(
                f'<span style="color:#475569;">Step {i+1:>2}</span>'
                f' <span style="color:#334155;">│</span>'
                f' <span style="color:#94a3b8;">{act:<22}</span>'
                f' <span style="color:#334155;">│</span>'
                f' <span style="color:{color};">reward: {sign}{r:.4f}</span>'
            )
            if specs:
                lines.append(f'<span style="color:#334155;">         │  → called: <span style="color:#7c3aed;">{html.escape(", ".join(specs))}</span></span>')
            if mode:
                lines.append(f'<span style="color:#334155;">         │  → mode:   <span style="color:#f59e0b;">{html.escape(mode)}</span></span>')
        total = sum(rewards)
        sign = "+" if total >= 0 else ""
        lines.append(f'<span style="color:#334155;">{"─"*56}</span>')
        lines.append(f'<span style="color:#e2e8f0;font-weight:600;">Total: {sign}{total:.4f}</span>'
                     f' <span style="color:#475569;">│ Steps: {len(rewards)}</span>')
        body = "\n".join(lines)

    return (
        f'<div style="background:rgba(0,0,0,0.35);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:12px;padding:14px 16px;font-family:\'JetBrains Mono\',\'Fira Code\',monospace;'
        f'font-size:11.5px;line-height:1.8;min-height:200px;max-height:340px;overflow-y:auto;">'
        f'{body}</div>'
    )


# ─────────────────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────────────────

def do_reset(task_choice, custom_task, phase, progress=gr.Progress(track_tqdm=False)):
    progress(0, desc="Loading environment… (first run may take ~30s)")
    _, info = S.reset(int(phase))
    obs_dim  = int(S.env.observation_space.shape[0])
    act_dim  = int(S.env.action_space.shape[0])
    progress(1.0, desc="Ready")
    status   = f'Episode started  |  Task: "{S.task[:100]}"'
    return (
        status,
        _metrics(obs_dim, act_dim, S.registry.size, int(phase)),
        fig_reward_curve([]),
        fig_delegation_graph([], []),
        fig_reward_breakdown({}),
        _log_html([], []),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def do_step(action_type, specialist_choice):
    if S.env is None or S.done:
        return ("No active episode — reset first.",
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.update(interactive=False), gr.update(interactive=False))

    action = np.zeros(S.env.action_space.shape, dtype=np.float32)
    if action_type == "STOP":
        action[0] = 1.0
    elif action_type == "CALL SPECIALIST":
        action[0] = 0.0
        ids = S.registry.list_ids()
        if specialist_choice in ids:
            idx = ids.index(specialist_choice)
            if idx < S.env.max_specialists:
                action[1 + idx] = 1.0
        else:
            action[1] = 1.0
    elif action_type == "PARALLEL SPAWN":
        action[0] = 6.0
        action[1] = 1.0
        if S.env.max_specialists > 1:
            action[2] = 1.0
        action[1 + S.env.max_specialists] = 1.0
    else:
        action = S.env.action_space.sample()

    _, r, term, trunc, info = S.step(action)
    done = term or trunc

    called = info.get("called_specialists", [])
    edges  = [(e.caller_id, e.callee_id) for e in S.env.delegation_graph.get_delegation_path()]
    sign   = "+" if r >= 0 else ""
    status = f"Step {S.step_n}  |  reward {sign}{r:.4f}  |  {'DONE' if done else 'Running…'}"
    if done:
        status += f"  |  Total: {sum(S.rewards):+.4f}"

    return (
        status,
        fig_reward_curve(S.rewards),
        fig_delegation_graph(called, edges),
        fig_reward_breakdown(info.get("reward_components", {})),
        _log_html(S.actions, S.rewards),
        gr.update(interactive=not done),
        gr.update(interactive=not done),
    )


def do_run_full(task_choice, custom_task, phase, progress=gr.Progress(track_tqdm=False)):
    progress(0, desc="Loading environment…")
    S.reset(int(phase))
    progress(0.1, desc="Running episode…")
    info = {}
    for _ in range(15):
        if S.done:
            break
        _, _, _, _, info = S.step(S.env.action_space.sample())

    called = info.get("called_specialists", []) if info else []
    edges  = [(e.caller_id, e.callee_id) for e in S.env.delegation_graph.get_delegation_path()]
    obs_dim = int(S.env.observation_space.shape[0])
    act_dim = int(S.env.action_space.shape[0])
    total   = sum(S.rewards)
    status  = f"Episode complete  |  {S.step_n} steps  |  Total reward: {total:+.4f}"

    return (
        status,
        _metrics(obs_dim, act_dim, S.registry.size, int(phase)),
        fig_reward_curve(S.rewards),
        fig_delegation_graph(called, edges),
        fig_reward_breakdown(info.get("reward_components", {}) if info else {}),
        _log_html(S.actions, S.rewards),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def do_add_specialist(sid, role, desc, sim_plot_state):
    if not (sid.strip() and role.strip() and desc.strip()):
        return "Fill in all three fields.", sim_plot_state
    try:
        S.boot()
        S.registry.add_specialist({
            "id": sid.strip(), "role": role.strip(), "description": desc.strip(),
            "complexity_affinity": ["moderate", "complex"],
            "avg_latency_ms": 5000,
        })
        return (
            f"'{sid.strip()}' added. Policy can represent it via its 384-dim embedding — no retraining needed.",
            fig_similarity(S.registry),
        )
    except Exception as e:
        return f"Error: {e}", sim_plot_state


def do_load_demo():
    p = ASSETS / "demo_moment_1.json"
    if not p.exists():
        msg = '<div style="color:#ef4444;padding:20px;">Run <code>python demo/precompute_demo.py</code> first.</div>'
        return msg, msg
    with open(p) as f:
        d = json.load(f)

    def box(label, color, text):
        return (
            f'<div style="background:{color}08;border:1px solid {color}25;border-radius:12px;padding:18px;">'
            f'<div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:10px;">{label}</div>'
            f'<pre style="font-size:11.5px;color:#94a3b8;white-space:pre-wrap;'
            f'font-family:inherit;margin:0;line-height:1.6;">{html.escape(text[:700])}</pre></div>'
        )
    return (
        box("Generalist Output (No Delegation)", "#ef4444", d["generalist_output"]),
        box("Specialist-Routed Output (Learned Policy)", "#10b981", d["specialist_output"]),
    )


def do_reward_lab(lw, ep, fp, cw, eb):
    comps = {
        "quality_delta":      0.42,
        "efficiency_penalty": -ep * 2,
        "failure_penalty":    -fp * 0.3,
        "recovery_bonus":     0.08,
        "conflict_penalty":   -0.05,
        "conflict_bonus":     0.03,
        "consistency_bonus":  cw * 0.6,
        "latency_penalty":    -lw * 0.25,
        "explanation_bonus":  eb,
    }
    total = sum(comps.values())
    sign  = "+" if total >= 0 else ""
    summary = (
        f'<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.18);'
        f'border-radius:10px;padding:14px 18px;font-size:13px;color:#94a3b8;">'
        f'Estimated total reward: <span style="color:#00d4ff;font-weight:700;font-size:18px;">'
        f'{sign}{total:.3f}</span></div>'
    )
    return fig_reward_breakdown(comps), summary


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background:#0f0f1a !important; font-family:'Inter',system-ui,sans-serif !important; }
.gr-button { border-radius:8px !important; font-weight:600 !important; font-size:13px !important; transition:all .2s !important; }
.gr-button-primary {
  background:linear-gradient(135deg,#00d4ff,#0092bb) !important;
  border:none !important; color:#0a0f1a !important;
}
.gr-button-primary:hover { transform:translateY(-1px) !important; box-shadow:0 4px 18px rgba(0,212,255,0.35) !important; }
.gr-button-secondary {
  background:rgba(255,255,255,0.04) !important;
  border:1px solid rgba(255,255,255,0.09) !important; color:#e2e8f0 !important;
}
.gr-button-secondary:hover { background:rgba(255,255,255,0.07) !important; }
.gr-form, .gr-box, .gr-panel {
  background:rgba(255,255,255,0.025) !important;
  border:1px solid rgba(255,255,255,0.08) !important; border-radius:12px !important;
}
label { color:#475569 !important; font-size:11px !important; font-weight:600 !important;
        text-transform:uppercase !important; letter-spacing:.6px !important; }
input, textarea, select {
  background:rgba(0,0,0,0.3) !important; border:1px solid rgba(255,255,255,0.08) !important;
  color:#e2e8f0 !important; border-radius:8px !important;
}
.tabitem { background:transparent !important; }
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:4px; }
::-webkit-scrollbar-track { background:transparent; }
"""

# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────

def _load_catalog_yaml() -> list[dict]:
    """Load specialist data directly from YAML (no embeddings, instant)."""
    import yaml
    with open(CATALOG) as f:
        return yaml.safe_load(f)["specialists"]


def _spec_cards_from_yaml(specialists: list[dict]) -> str:
    cards = ""
    for sp in specialists:
        c = SPEC_COLORS.get(sp["id"], "#7c3aed")
        desc = html.escape(sp["description"][:88])
        cards += f"""
<div style="background:rgba(255,255,255,0.025);border:1px solid {c}18;border-left:3px solid {c};
            border-radius:12px;padding:14px;transition:all .2s;">
  <div style="font-size:11px;font-weight:700;color:{c};margin-bottom:6px;">
    <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
                 background:{c};box-shadow:0 0 6px {c}80;margin-right:5px;"></span>
    {sp['role']}
  </div>
  <div style="font-size:11px;color:#64748b;line-height:1.5;">{desc}…</div>
  <div style="font-size:10px;color:#334155;margin-top:8px;padding-top:8px;
              border-top:1px solid rgba(255,255,255,0.05);">
    {sp['avg_latency_ms']}ms avg &nbsp;·&nbsp; {', '.join(sp['complexity_affinity'])}
  </div>
</div>"""
    return f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0;">{cards}</div>'


def build():
    # Load catalog from YAML only — no embeddings, instant startup
    catalog = _load_catalog_yaml()
    n_spec = len(catalog)
    obs0 = EpisodeState.observation_dim(6)   # 6 = default max_specialists
    act0 = 6 + 6                              # max_specialists(6) + 6

    with gr.Blocks(title="SpindleFlow RL") as app:

        gr.HTML(_hero())

        with gr.Tabs():

            # ══════════════════════════════════════════════
            # TAB 1  Live Demo
            # ══════════════════════════════════════════════
            with gr.Tab("Live Demo"):
                metrics_box = gr.HTML(_metrics(obs0, act0, n_spec, 1))

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(_sec("Task"))
                        task_dd  = gr.Dropdown(choices=PRESET_TASKS, value=PRESET_TASKS[0], label="Preset task")
                        task_txt = gr.Textbox(label="Or enter custom task", placeholder="Describe a software engineering task…")
                        phase_sl = gr.Slider(1, 3, value=1, step=1, label="Curriculum phase")

                    with gr.Column(scale=2):
                        gr.HTML(_sec("Controls"))
                        reset_btn = gr.Button("Reset Episode", variant="primary", size="lg")
                        run_btn   = gr.Button("Run Full Episode", variant="secondary", size="lg")
                        gr.HTML('<div style="height:8px;"></div>')
                        act_dd  = gr.Dropdown(
                            choices=["RANDOM", "STOP", "CALL SPECIALIST", "PARALLEL SPAWN"],
                            value="RANDOM", label="Action type",
                        )
                        _spec_ids = [sp["id"] for sp in catalog]
                        spec_dd = gr.Dropdown(choices=_spec_ids, value=_spec_ids[0],
                                              label="Target specialist")
                        step_btn = gr.Button("Execute One Step", variant="secondary", interactive=False)

                status_box = gr.Textbox(label="Status", value="Click 'Reset Episode' to start.",
                                        interactive=False, lines=1)

                with gr.Row():
                    reward_plot = gr.Plot(value=fig_reward_curve([]),   label="")
                    graph_plot  = gr.Plot(value=fig_delegation_graph([], []), label="")

                with gr.Row():
                    breakdown_plot = gr.Plot(value=fig_reward_breakdown({}), label="")
                    log_box = gr.HTML(_log_html([], []))

                # Wiring
                common_outs = [status_box, metrics_box, reward_plot, graph_plot,
                               breakdown_plot, log_box, step_btn, run_btn, reset_btn]

                reset_btn.click(do_reset,
                                inputs=[task_dd, task_txt, phase_sl],
                                outputs=common_outs)

                step_btn.click(do_step,
                               inputs=[act_dd, spec_dd],
                               outputs=[status_box, reward_plot, graph_plot,
                                        breakdown_plot, log_box, step_btn, run_btn])

                run_btn.click(do_run_full,
                              inputs=[task_dd, task_txt, phase_sl],
                              outputs=common_outs)

            # ══════════════════════════════════════════════
            # TAB 2  Specialist Roster
            # ══════════════════════════════════════════════
            with gr.Tab("Specialists"):
                gr.HTML(_sec("Roster (8 specialists, capability-embedded)"))
                gr.HTML(_spec_cards_from_yaml(catalog))

                gr.HTML(_sec("Capability Similarity Matrix"))
                sim_load_btn = gr.Button("Load Similarity Matrix", variant="secondary")
                sim_plot = gr.Plot(value=None, label="")

                gr.HTML(_sec("Add Specialist Dynamically"))
                gr.HTML('<div style="font-size:12px;color:#475569;margin-bottom:12px;">'
                        'New specialists are immediately representable via their 384-dim embedding — '
                        'no retraining or YAML edits required.</div>')
                with gr.Row():
                    new_id   = gr.Textbox(label="ID",   placeholder="ml_engineer")
                    new_role = gr.Textbox(label="Role", placeholder="ML Engineer")
                new_desc = gr.Textbox(label="Description",
                                      placeholder="Expert in PyTorch, model training, MLOps pipelines…",
                                      lines=2)
                with gr.Row():
                    add_btn = gr.Button("Add to Roster", variant="primary")
                add_status = gr.Textbox(label="Result", interactive=False)

                def load_sim():
                    S.boot()
                    return fig_similarity(S.registry)

                sim_load_btn.click(fn=load_sim, outputs=sim_plot)

                add_btn.click(do_add_specialist,
                              inputs=[new_id, new_role, new_desc, sim_plot],
                              outputs=[add_status, sim_plot])

            # ══════════════════════════════════════════════
            # TAB 3  Training
            # ══════════════════════════════════════════════
            with gr.Tab("Training"):
                gr.HTML(_sec("Simulated Training Curve"))
                gr.Plot(value=fig_training_curve(), label="")

                gr.HTML(_sec("Curriculum Phases"))
                gr.HTML("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;">
  <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.18);border-radius:12px;padding:18px;">
    <div style="font-size:10px;font-weight:700;color:#00d4ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Phase 1 · Atomic/Simple</div>
    <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:5px;">200 episodes</div>
    <div style="font-size:11px;color:#475569;">Agent learns basic routing — which single specialist to call.</div>
  </div>
  <div style="background:rgba(124,58,237,0.04);border:1px solid rgba(124,58,237,0.18);border-radius:12px;padding:18px;">
    <div style="font-size:10px;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Phase 2 · Moderate</div>
    <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:5px;">400 episodes</div>
    <div style="font-size:11px;color:#475569;">Agent learns multi-specialist coordination and mode selection.</div>
  </div>
  <div style="background:rgba(245,158,11,0.04);border:1px solid rgba(245,158,11,0.18);border-radius:12px;padding:18px;">
    <div style="font-size:10px;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Phase 3 · Complex/Enterprise</div>
    <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:5px;">600 episodes</div>
    <div style="font-size:11px;color:#475569;">Full delegation strategy with DAG depth, fallbacks, and latency trade-offs.</div>
  </div>
</div>""")

                gr.HTML(_sec("Quick Start Commands"))
                with gr.Row():
                    gr.Code(value=(
                        "# Demo mode (no OpenAI needed)\n"
                        "cd spindleflow-rl\n"
                        "python training/train.py \\\n"
                        "  --phase 1 \\\n"
                        "  --timesteps 50000 \\\n"
                        "  --demo-mode\n\n"
                        "# Watch curves\n"
                        "tensorboard --logdir tensorboard_logs/"
                    ), language="python", label="Local")
                    gr.Code(value=(
                        "# Google Colab (T4 GPU, free)\n"
                        "!git clone https://github.com/YOUR/spindleflow-rl\n"
                        "%cd spindleflow-rl\n"
                        "!pip install -r requirements.txt sb3-contrib\n\n"
                        "# 5k-step demo run\n"
                        "%run colab/train_colab.py"
                    ), language="python", label="Colab")

            # ══════════════════════════════════════════════
            # TAB 4  Quality Demo
            # ══════════════════════════════════════════════
            with gr.Tab("Quality Demo"):
                gr.HTML(_sec("Before vs After Delegation Learning"))
                load_btn = gr.Button("Load Demo Comparison", variant="primary")
                with gr.Row():
                    gen_html  = gr.HTML()
                    spec_html = gr.HTML()
                load_btn.click(do_load_demo, outputs=[gen_html, spec_html])

                gr.HTML(_sec("Policy Tuning — Quality vs Latency"))
                gr.Plot(value=fig_policy_compare(), label="")
                gr.HTML("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;">
  <div style="background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.2);border-radius:12px;padding:16px;">
    <div style="font-size:10px;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Quality Policy</div>
    <div style="font-size:11px;color:#64748b;line-height:1.7;">5 specialists · sequential · ~180s<br>
    <code style="color:#a78bfa;background:rgba(124,58,237,0.1);padding:1px 5px;border-radius:4px;">latency_weight=0.0</code></div>
  </div>
  <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);border-radius:12px;padding:16px;">
    <div style="font-size:10px;font-weight:700;color:#00d4ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Latency Policy</div>
    <div style="font-size:11px;color:#64748b;line-height:1.7;">3 specialists · parallel · ~45s<br>
    <code style="color:#00d4ff;background:rgba(0,212,255,0.1);padding:1px 5px;border-radius:4px;">latency_weight=0.15</code></div>
  </div>
</div>""")

            # ══════════════════════════════════════════════
            # TAB 5  Reward Lab
            # ══════════════════════════════════════════════
            with gr.Tab("Reward Lab"):
                gr.HTML(_sec("Interactive Reward Explorer"))
                gr.HTML('<div style="font-size:12px;color:#475569;margin-bottom:16px;">'
                        'Tune the reward weights and see how each component contributes to the total signal.</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        s_lw = gr.Slider(0.0, 0.5,  value=0.05, step=0.01, label="Latency Weight")
                        s_ep = gr.Slider(0.0, 0.2,  value=0.05, step=0.01, label="Efficiency Penalty")
                        s_fp = gr.Slider(0.0, 1.0,  value=0.30, step=0.05, label="Failure Penalty")
                        s_cw = gr.Slider(0.0, 0.5,  value=0.10, step=0.01, label="Consistency Bonus")
                        s_eb = gr.Slider(0.0, 0.2,  value=0.05, step=0.01, label="Explanation Bonus")
                    with gr.Column(scale=2):
                        lab_plot    = gr.Plot(label="")
                        lab_summary = gr.HTML()

                sliders = [s_lw, s_ep, s_fp, s_cw, s_eb]
                for sl in sliders:
                    sl.change(do_reward_lab, inputs=sliders, outputs=[lab_plot, lab_summary])
                app.load(do_reward_lab, inputs=sliders, outputs=[lab_plot, lab_summary])

            # ══════════════════════════════════════════════
            # TAB 6  Architecture
            # ══════════════════════════════════════════════
            with gr.Tab("Architecture"):
                gr.HTML(f"""
{_sec("System Design")}
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;">

  <div style="background:rgba(0,212,255,0.03);border:1px solid rgba(0,212,255,0.14);border-radius:12px;padding:18px;">
    <div style="font-size:10px;font-weight:700;color:#00d4ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Observation Space ({obs0:,}-dim flat vector)</div>
    <table style="font-size:11.5px;color:#64748b;width:100%;border-collapse:collapse;">
      <tr><td style="color:#e2e8f0;padding:3px 0;width:50px;">384</td><td>Task embedding (all-MiniLM-L6-v2)</td></tr>
      <tr><td style="color:#e2e8f0;">2304</td><td>Roster embeddings (6 × 384)</td></tr>
      <tr><td style="color:#e2e8f0;">2304</td><td>Called embeddings (6 × 384)</td></tr>
      <tr><td style="color:#e2e8f0;">384</td><td>Scratchpad embedding</td></tr>
      <tr><td style="color:#e2e8f0;">100</td><td>Delegation graph adj. (10×10)</td></tr>
      <tr><td style="color:#e2e8f0;">6</td><td>Called specialist mask</td></tr>
      <tr><td style="color:#e2e8f0;">8</td><td>Scalar features</td></tr>
    </table>
  </div>

  <div style="background:rgba(124,58,237,0.03);border:1px solid rgba(124,58,237,0.14);border-radius:12px;padding:18px;">
    <div style="font-size:10px;font-weight:700;color:#a78bfa;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Action Space ({act0}-dim Box)</div>
    <table style="font-size:11.5px;color:#64748b;width:100%;border-collapse:collapse;">
      <tr><td style="color:#e2e8f0;padding:3px 0;width:50px;">[0]</td><td>Meta-action (STOP / CALL / PARALLEL…)</td></tr>
      <tr><td style="color:#e2e8f0;">[1:7]</td><td>Specialist selection logits (multi-hot)</td></tr>
      <tr><td style="color:#e2e8f0;">[7]</td><td>Delegation mode (SEQ / PAR / FAN-OUT…)</td></tr>
      <tr><td style="color:#e2e8f0;">[8:12]</td><td>Mode parameters (rounds, threshold…)</td></tr>
    </table>
  </div>
</div>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px;">
  <div style="background:rgba(16,185,129,0.03);border:1px solid rgba(16,185,129,0.14);border-radius:12px;padding:16px;">
    <div style="font-size:10px;font-weight:700;color:#34d399;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Policy</div>
    <div style="font-size:11.5px;color:#64748b;line-height:1.8;">LSTM PPO (RecurrentPPO)<br>MlpLstmPolicy<br>Hidden: 256 · 1 layer<br>POMDP-safe via LSTM state<br>4 factored action heads</div>
  </div>
  <div style="background:rgba(245,158,11,0.03);border:1px solid rgba(245,158,11,0.14);border-radius:12px;padding:16px;">
    <div style="font-size:10px;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Tiered Reward</div>
    <div style="font-size:11.5px;color:#64748b;line-height:1.8;">T0 — Structural heuristics<br>T1 — Cosine embedding sim<br>T2 — GPT-4o-mini judge<br>T3 — Full judge (ckpts)<br>Episode-level tier lock</div>
  </div>
  <div style="background:rgba(239,68,68,0.03);border:1px solid rgba(239,68,68,0.14);border-radius:12px;padding:16px;">
    <div style="font-size:10px;font-weight:700;color:#f87171;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Safety</div>
    <div style="font-size:11.5px;color:#64748b;line-height:1.8;">DAG cycle detection (DFS)<br>Max delegation depth: 2<br>Scratchpad sandbox isolation<br>Injection sanitization<br>Action masking (DAG)</div>
  </div>
</div>

<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px;">
  <div style="font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Reward Function</div>
  <pre style="font-size:12px;color:#94a3b8;line-height:1.9;margin:0;font-family:'JetBrains Mono','Fira Code',monospace;"><span style="color:#e2e8f0;">total_reward</span> = (
  quality_delta          <span style="color:#334155;"># specialist_score − baseline  (same tier)</span>
− efficiency_penalty     <span style="color:#334155;"># 0.05 × max(0, n_called − expected)</span>
− failure_penalty        <span style="color:#334155;"># 0.3 per timeout, 0.2 per error</span>
+ recovery_bonus         <span style="color:#334155;"># +0.1 if fallback succeeded</span>
− conflict_penalty       <span style="color:#334155;"># 0.1 per unresolved conflict</span>
+ conflict_bonus         <span style="color:#334155;"># 0.05 per resolved conflict</span>
+ consistency_bonus      <span style="color:#334155;"># 0.1 × Dirichlet-prior path score</span>
− latency_penalty        <span style="color:#334155;"># latency_weight × overage_fraction</span>
+ explanation_bonus      <span style="color:#334155;"># 0.05 if delegation is auditable</span>
)</pre>
</div>
""")

    return app


_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui"],
)

if __name__ == "__main__":
    print("Booting SpindleFlow RL Dashboard…")
    print("Background pre-warm started (sentence-transformer). UI will be ready immediately.")
    demo = build()
    demo.queue(max_size=4)
    demo.launch(
        server_name="0.0.0.0", server_port=7860,
        share=False, show_error=True,
        theme=_THEME, css=CSS,
    )
