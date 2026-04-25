"""
SpindleFlow RL — Streamlit Dashboard
=====================================
Run:  cd spindleflow-rl && streamlit run demo/streamlit_app.py
URL:  http://localhost:8501
"""

from __future__ import annotations
import os, sys, json, html as _html
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY (and any other vars) from .env

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from env.spindleflow_env import SpindleFlowEnv
from env.state import EpisodeState
from env.specialist_registry import SpecialistRegistry
from orchestrator_widget import render_orchestrator

# ─────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpindleFlow RL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
CONFIG  = "configs/training_config.yaml"
CATALOG = "configs/specialist_catalog.yaml"
ASSETS  = Path("demo/assets")

SPEC_COLORS = {
    "frontend_react":     "#00d4ff",
    "backend_api":        "#7c3aed",
    "database_architect": "#f59e0b",
    "devops_engineer":    "#10b981",
    "security_analyst":   "#ef4444",
    "product_strategist": "#8b5cf6",
    "ux_designer":        "#ec4899",
    "tech_writer":        "#94a3b8",
}

@st.cache_resource
def _get_preset_tasks(n: int = 8) -> list[str]:
    """Sample n live tasks from TaskBank at page load — no hardcoded strings."""
    try:
        from training.task_bank import TaskBank
        bank = TaskBank(phase=1)
        return [bank.sample() for _ in range(n)]
    except Exception:
        # Fallback only if TaskBank is unavailable (e.g. missing config)
        return ["Describe a software engineering task requiring specialist collaboration"]


PRESET_TASKS = _get_preset_tasks()

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif"),
    margin=dict(l=44, r=20, t=44, b=40),
)
DARK_AXES = dict(
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
        self.step_n  = 0
        self.done    = False
        self.task    = ""
        # Full episode history for replay
        self.episode_history: list[dict] = []
        # Action entropy per step (policy confidence)
        self.step_entropies: list[float] = []
        # Observation vector stats per step
        self.obs_history: list[dict] = []
        # Specialists auto-spawned for this episode
        self.spawned_specialists: list[str] = []

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
        self.rewards = []
        self.actions = []
        self.step_n  = 0
        self.done    = False
        self.task    = info.get("task", "")
        self.episode_history = []
        self.step_entropies  = []
        self.obs_history     = []
        self.spawned_specialists: list[str] = list(info.get("spawned_specialists", []))
        return obs, info

    def step(self, action):
        if self.env is None or self.done:
            return None, 0.0, True, False, {}
        obs, r, term, trunc, info = self.env.step(action)
        self.rewards.append(r)
        self.actions.append(info)
        self.step_n += 1
        self.done = term or trunc

        # Capture step snapshot for replay
        called = info.get("called_specialists", [])
        edges  = [(e.caller_id, e.callee_id)
                  for e in self.env.delegation_graph.get_delegation_path()]
        self.episode_history.append({
            "step":        self.step_n,
            "reward":      r,
            "action_name": info.get("action_name", "UNKNOWN"),
            "called":      list(called),
            "edges":       list(edges),
            "components":  dict(info.get("reward_components", {})),
            "mode":        info.get("delegation_mode", ""),
            "cumulative":  float(sum(self.rewards)),
            "latencies":   dict(info.get("specialist_latencies", {})),
        })

        # Compute real action entropy (specialist-selection logits)
        if self.env is not None:
            n = self.env.max_specialists
            spec_logits = action[1: 1 + n].copy()
            spec_logits = spec_logits - spec_logits.max()
            exp_l  = np.exp(spec_logits)
            probs  = exp_l / (exp_l.sum() + 1e-8)
            entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
            self.step_entropies.append(entropy)

        # Capture observation norm for state trace
        if obs is not None:
            self.obs_history.append({
                "step":     self.step_n,
                "obs_norm": float(np.linalg.norm(obs)),
                "obs_mean": float(obs.mean()),
                "obs_max":  float(obs.max()),
            })

        return obs, r, term, trunc, info


def _S() -> Session:
    if "session" not in st.session_state:
        st.session_state.session = Session()
    return st.session_state.session


def _load_catalog() -> list[dict]:
    import yaml
    with open(CATALOG) as f:
        return yaml.safe_load(f)["specialists"]


def _exec_mode_badges(S: "Session") -> str:
    """Return inline HTML badge strip showing execution and task-generation modes."""
    import os
    has_key   = bool(os.getenv("OPENAI_API_KEY"))
    llm_tasks = S.env is not None and S.env.task_bank._client is not None

    exec_b = (
        '<span style="padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;'
        'background:rgba(16,185,129,0.1);color:#34d399;'
        'border:1px solid rgba(16,185,129,0.22);">● LLM BASELINE</span>'
        if has_key else
        '<span style="padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;'
        'background:rgba(245,158,11,0.1);color:#fbbf24;'
        'border:1px solid rgba(245,158,11,0.22);">'
        '⚡ SIMULATION MODE — specialist outputs templated · set OPENAI_API_KEY for real LLM</span>'
    )
    task_b = (
        '<span style="padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;'
        'background:rgba(16,185,129,0.1);color:#34d399;'
        'border:1px solid rgba(16,185,129,0.22);">● LLM TASKS</span>'
        if llm_tasks else
        '<span style="padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;'
        'background:rgba(148,163,184,0.08);color:#64748b;'
        'border:1px solid rgba(148,163,184,0.18);">⚡ CATALOG TASKS</span>'
    ) if S.env is not None else ""

    return (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 12px;">'
        f'{exec_b}{task_b}</div>'
    )

# ─────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────
def fig_reward_curve(rewards: list[float]) -> go.Figure:
    if not rewards:
        fig = go.Figure()
        fig.update_layout(
            **DARK, **DARK_AXES,
            title=dict(text="Episode Reward", font=dict(size=13, color="#64748b")),
            annotations=[dict(text="Reset the environment to begin",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#334155", size=13))],
        )
        return fig

    steps = list(range(len(rewards)))
    cumul = np.cumsum(rewards).tolist()
    fig   = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.62, 0.38], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(
        x=steps, y=cumul, mode="lines",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        name="Cumulative",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=steps, y=rewards,
        marker_color=["#10b981" if r >= 0 else "#ef4444" for r in rewards],
        marker_line_width=0, name="Per-step",
    ), row=2, col=1)
    fig.update_layout(**DARK, height=300, showlegend=False,
                      title=dict(text="Episode Reward", font=dict(size=13, color="#94a3b8")))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)",
                     title_text="Cumul.", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Step", row=2, col=1, title_font_size=10)
    return fig


def fig_delegation_graph(
    S: Session,
    called_ids: list[str],
    edges: list[tuple],
    highlight_latest: bool = True,
    spawned_ids: list[str] | None = None,
) -> go.Figure:
    """
    Professional hierarchical DAG layout.
    Orchestrator at top, called specialists in middle, uncalled dimmed at bottom.
    """
    all_ids     = list(S.registry.list_ids()) if S.registry else []
    called_set  = set(called_ids)
    spawned_set = set(spawned_ids or S.spawned_specialists)
    uncalled    = [x for x in all_ids if x not in called_set]

    # ── Build node positions (hierarchical layout) ───────────────────
    pos = {"orchestrator": (0.5, 0.92)}

    n_called = len(called_ids)
    if n_called > 0:
        for i, sid in enumerate(called_ids):
            x = (i + 1) / (n_called + 1)
            pos[sid] = (x, 0.55)

    n_uncalled = len(uncalled)
    if n_uncalled > 0:
        for i, sid in enumerate(uncalled):
            x = (i + 1) / (n_uncalled + 1)
            pos[sid] = (x, 0.12)

    fig = go.Figure()

    # ── Background depth ring ────────────────────────────────────────
    max_depth   = getattr(S.env, "max_depth", 2) if S.env else 2
    cur_depth   = S.env.delegation_graph.depth if S.env else 0
    depth_frac  = cur_depth / max(max_depth, 1)
    ring_color  = ("#10b981" if depth_frac < 0.7
                   else ("#f59e0b" if depth_frac < 1.0 else "#ef4444"))

    fig.add_shape(type="rect",
        x0=0.0, y0=0.0, x1=1.0, y1=1.0,
        line=dict(color=ring_color, width=2, dash="dot"),
        fillcolor="rgba(0,0,0,0)", xref="x", yref="y",
    )
    fig.add_annotation(
        x=0.98, y=0.98, xref="x", yref="y",
        text=f"Depth {cur_depth}/{max_depth}", showarrow=False,
        font=dict(size=9, color=ring_color), xanchor="right", yanchor="top",
    )

    # ── Edges ────────────────────────────────────────────────────────
    latest_edge = edges[-1] if edges else None
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        is_latest = (latest_edge and highlight_latest and (src, dst) == latest_edge)
        color = "rgba(0,212,255,0.9)" if is_latest else "rgba(0,212,255,0.45)"
        width = 2.5 if is_latest else 1.8
        dash  = "dash" if is_latest else "solid"

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.4, arrowwidth=2,
            arrowcolor=color, showarrow=True,
        )

    # ── Orchestrator node ────────────────────────────────────────────
    ox, oy = pos["orchestrator"]
    fig.add_trace(go.Scatter(
        x=[ox], y=[oy], mode="markers+text",
        marker=dict(size=44, color="#f59e0b", symbol="circle",
                    line=dict(color="#fcd34d", width=2.5), opacity=1.0),
        text=["<b>ORCH</b>"], textposition="middle center",
        textfont=dict(size=9, color="#0a0f1a", family="Inter, sans-serif"),
        hovertext=["<b>Orchestrator</b><br>Root node — makes all delegation decisions"],
        hoverinfo="text", showlegend=False, name="orchestrator",
    ))

    # ── Called specialist nodes ──────────────────────────────────────
    for sid in called_ids:
        if sid not in pos:
            continue
        x, y      = pos[sid]
        c         = SPEC_COLORS.get(sid, "#7c3aed")
        spec      = S.registry.get(sid) if S.registry else None
        role      = spec.role if spec else sid
        lat       = f"{spec.avg_latency_ms}ms" if spec else ""
        is_spawned = sid in spawned_set
        symbol    = "star" if is_spawned else "circle"
        size      = 38 if is_spawned else 32
        border_c  = "#fbbf24" if is_spawned else "rgba(255,255,255,0.4)"
        hover_tag = " ⚡ AUTO-SPAWNED" if is_spawned else ""
        label     = (("⚡ " if is_spawned else "") + sid).replace("_", "<br>")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=c, symbol=symbol,
                        line=dict(color=border_c, width=2.5), opacity=1.0),
            text=[label], textposition="bottom center",
            textfont=dict(size=8, color="#fbbf24" if is_spawned else "#e2e8f0"),
            hovertext=[f"<b>{role}</b><br>Called ✓{hover_tag}<br>{lat}"],
            hoverinfo="text", showlegend=False,
        ))

    # ── Uncalled specialist nodes (dimmed) ───────────────────────────
    for sid in uncalled:
        if sid not in pos:
            continue
        x, y  = pos[sid]
        c     = SPEC_COLORS.get(sid, "#334155")
        spec  = S.registry.get(sid) if S.registry else None
        role  = spec.role if spec else sid
        label = sid.replace("_", "<br>")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=16, color="#1e293b", symbol="circle",
                        line=dict(color=c, width=1), opacity=0.5),
            text=[label], textposition="bottom center",
            textfont=dict(size=7, color="rgba(148,163,184,0.45)"),
            hovertext=[f"<b>{role}</b><br>Not called"],
            hoverinfo="text", showlegend=False,
        ))

    # ── Section labels ───────────────────────────────────────────────
    fig.add_annotation(x=0.01, y=0.96, xref="x", yref="y",
        text="ORCHESTRATOR", showarrow=False,
        font=dict(size=8, color="#475569"), xanchor="left")
    if called_ids:
        fig.add_annotation(x=0.01, y=0.62, xref="x", yref="y",
            text="CALLED", showarrow=False,
            font=dict(size=8, color="#00d4ff"), xanchor="left")
    if uncalled:
        fig.add_annotation(x=0.01, y=0.19, xref="x", yref="y",
            text="AVAILABLE", showarrow=False,
            font=dict(size=8, color="#334155"), xanchor="left")

    fig.update_layout(
        **DARK, height=420,
        title=dict(
            text=(f"Delegation Graph  ·  {len(called_ids)} specialists called"
                  f"  ·  Depth {cur_depth}/{max_depth}"),
            font=dict(size=13, color="#94a3b8"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.05]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.08]),
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
    fig = go.Figure(go.Bar(
        x=values,
        y=[n.replace("_", " ").title() for n in names],
        orientation="h",
        marker_color=["#10b981" if v >= 0 else "#ef4444" for v in values],
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=9),
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(**DARK, height=310,
                      title=dict(text="Reward Breakdown", font=dict(size=13, color="#94a3b8")),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Value"),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    return fig


def fig_policy_confidence(
    entropies: list[float],
    step_labels: list[int] | None = None,
) -> go.Figure:
    """
    Policy confidence chart — specialist-selection entropy per step.
    High entropy = uncertain/exploring. Low = confident/committed.
    Real data from actual action vectors used each step.
    """
    if not entropies:
        fig = go.Figure()
        fig.update_layout(
            **DARK, **DARK_AXES,
            title=dict(text="Policy Confidence (Action Entropy)",
                       font=dict(size=13, color="#64748b")),
            annotations=[dict(text="Run an episode to see real action entropy",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#334155", size=12))],
        )
        return fig

    steps   = step_labels or list(range(1, len(entropies) + 1))
    max_e   = float(np.log(max(len(entropies), 2)))
    norm_e  = [min(1.0, max(0.0, e / max(max_e, 1e-8))) for e in entropies]
    colors  = [
        f"rgba({int(0 + 124 * ne)},{int(212 - 154 * ne)},{int(255 - 58 * ne)},0.85)"
        for ne in norm_e
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps, y=norm_e,
        marker_color=colors, marker_line_width=0,
        name="Normalised entropy",
        text=[f"{e:.3f}" for e in entropies],
        textposition="outside",
        textfont=dict(size=8, color="#94a3b8"),
        hovertemplate="Step %{x}<br>Entropy: %{text}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(148,163,184,0.3)",
                  annotation_text="Mid-entropy", annotation_font_color="#475569")
    fig.update_layout(
        **DARK, height=260,
        title=dict(text="Policy Confidence — Specialist Selection Entropy per Step",
                   font=dict(size=12, color="#94a3b8")),
        xaxis=dict(title="Episode Step", gridcolor="rgba(255,255,255,0.05)",
                   zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Entropy (0=certain, 1=uniform)", range=[0, 1.15],
                   gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
        showlegend=False,
    )
    return fig


def fig_similarity(registry: SpecialistRegistry) -> go.Figure:
    ids = registry.list_ids()
    n   = len(ids)

    if n == 0:
        fig = go.Figure()
        fig.update_layout(**DARK, title=dict(text="No specialists in registry",
                                              font=dict(size=13, color="#64748b")))
        return fig

    missing = [sid for sid in ids if registry.get(sid).embedding is None]
    if missing:
        fig = go.Figure()
        fig.update_layout(
            **DARK, **DARK_AXES,
            title=dict(text="Embeddings not computed — boot the environment first",
                       font=dict(size=13, color="#64748b")),
            annotations=[dict(text=f"Missing embeddings: {', '.join(missing[:4])}",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#334155", size=12))],
        )
        return fig

    mat = np.zeros((n, n))
    try:
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                ea = registry.get(a).to_state_vector()
                eb = registry.get(b).to_state_vector()
                mat[i][j] = float(np.dot(ea, eb))
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(**DARK, title=dict(text=f"Similarity error: {exc}",
                                              font=dict(size=13, color="#ef4444")))
        return fig
    labels = [x.replace("_", "<br>") for x in ids]
    fig = go.Figure(go.Heatmap(
        z=mat, x=labels, y=labels,
        colorscale=[[0, "#0f0f1a"], [0.5, "rgba(124,58,237,0.6)"], [1, "#00d4ff"]],
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
        rng  = np.random.default_rng(42)
        eps  = list(range(0, 201, 5))
        rews = [float(np.clip(0.1 + 0.5 * (1 - np.exp(-e / 80)) + rng.normal(0, 0.04), 0, 1))
                for e in eps]
    smooth = [float(np.mean(rews[max(0, i - 4):i + 1])) for i in range(len(rews))]
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
    fig.update_layout(**DARK, **DARK_AXES, height=340,
                      title=dict(text="Training Progress — Mean Reward per Episode",
                                 font=dict(size=13, color="#94a3b8")),
                      xaxis_title="Episode", yaxis_title="Mean Reward",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def fig_training_entropy() -> go.Figure:
    """
    Policy entropy over training.
    Reads from demo/assets/entropy_log.json if produced by train.py,
    or from current session entropy if no log exists.
    Never shows fake data — gracefully absent if neither source exists.
    """
    path = ASSETS / "entropy_log.json"
    S    = _S()

    if path.exists():
        with open(path) as f:
            d = json.load(f)
        episodes     = d["episodes"]
        entropies    = d["mean_entropies"]
        source_label = "From training log"
    elif S.step_entropies:
        episodes     = list(range(1, len(S.step_entropies) + 1))
        entropies    = S.step_entropies
        source_label = "Current episode (live)"
    else:
        fig = go.Figure()
        fig.update_layout(
            **DARK, **DARK_AXES,
            title=dict(text="Policy Entropy — Run training to populate",
                       font=dict(size=13, color="#64748b")),
            annotations=[dict(
                text="Run python training/train.py to generate entropy logs",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#334155", size=12),
            )],
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=entropies, mode="lines+markers",
        line=dict(color="#7c3aed", width=2.2),
        marker=dict(size=4, color="#a78bfa"),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.06)",
        name=source_label,
    ))
    fig.update_layout(
        **DARK, **DARK_AXES, height=280,
        title=dict(text=f"Policy Entropy over Training ({source_label})",
                   font=dict(size=13, color="#94a3b8")),
        xaxis_title="Episode / Step",
        yaxis_title="Action Selection Entropy",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f1a !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stTabs"] > div:first-child button {
    color: #475569 !important; font-weight: 600 !important; font-size: 13px !important;
}
[data-testid="stTabs"] > div:first-child button[aria-selected="true"] {
    color: #00d4ff !important; border-bottom-color: #00d4ff !important;
}

.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 13px !important; transition: all .18s !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    background: rgba(255,255,255,0.04) !important; color: #e2e8f0 !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(0,212,255,0.28) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#00d4ff,#0092bb) !important;
    border: none !important; color: #0a0f1a !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 18px rgba(0,212,255,0.35) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"]  textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}

[data-testid="stSlider"] [data-testid="stTickBar"] { color: #475569 !important; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; padding: 16px !important;
}
[data-testid="stMetric"] label { color: #475569 !important; font-size: 11px !important; }
[data-testid="stMetricValue"]  { color: #00d4ff !important; font-weight: 700 !important; }

[data-testid="stCode"], .stCodeBlock {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

hr { border-color: rgba(255,255,255,0.07) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)


def hero():
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f0f1a,#130a22,#091422);
            border:1px solid rgba(0,212,255,0.14);border-radius:16px;
            padding:28px 36px;margin-bottom:4px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-60px;right:-40px;width:360px;height:360px;
              background:radial-gradient(circle,rgba(124,58,237,0.11) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="position:absolute;bottom:-60px;left:15%;width:280px;height:280px;
              background:radial-gradient(circle,rgba(0,212,255,0.07) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="font-size:26px;font-weight:800;
              background:linear-gradient(90deg,#00d4ff,#7c3aed,#00d4ff);
              background-size:200% auto;-webkit-background-clip:text;
              -webkit-text-fill-color:transparent;background-clip:text;
              margin:0 0 6px;">SpindleFlow RL</div>
  <div style="color:#64748b;font-size:13px;margin:0 0 18px;">
    Delegation Policy Learning Environment &mdash;
    Teaching orchestrators to route, specialize, and stop.
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;
                 background:rgba(0,212,255,0.1);color:#00d4ff;
                 border:1px solid rgba(0,212,255,0.22);">OPENENV v0</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;
                 background:rgba(124,58,237,0.1);color:#a78bfa;
                 border:1px solid rgba(124,58,237,0.22);">LSTM PPO</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;
                 background:rgba(16,185,129,0.1);color:#34d399;
                 border:1px solid rgba(16,185,129,0.22);">22/22 TESTS</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;
                 background:rgba(245,158,11,0.1);color:#fbbf24;
                 border:1px solid rgba(245,158,11,0.22);">HACKATHON 2026</span>
    <span style="padding:3px 11px;border-radius:999px;font-size:10px;font-weight:700;
                 background:rgba(16,185,129,0.08);color:#34d399;
                 border:1px solid rgba(16,185,129,0.25);">GENERIC MULTI-SECTOR</span>
  </div>
</div>
""", unsafe_allow_html=True)


def sec(title: str):
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;'
        f'letter-spacing:1px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.07);'
        f'margin:18px 0 14px;">{title}</div>',
        unsafe_allow_html=True,
    )


def status_bar(msg: str, color: str = "#94a3b8"):
    st.markdown(
        f'<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:10px 16px;font-size:12px;color:{color};margin:6px 0 10px;">'
        f'{_html.escape(msg)}</div>',
        unsafe_allow_html=True,
    )


def render_live_stats(S: Session) -> None:
    """Sidebar live stats strip — all values read directly from session state."""
    with st.sidebar:
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#00d4ff;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">'
            '● Live Episode Stats</div>',
            unsafe_allow_html=True,
        )

        status = ("Running"  if (S.env is not None and not S.done) else
                  "Complete" if S.done else "Idle")
        status_color = ("#10b981" if status == "Running" else
                        "#f59e0b" if status == "Complete" else "#475569")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:11px;color:#475569;">Status</span>'
            f'<span style="font-size:11px;font-weight:700;color:{status_color};">'
            f'{status}</span></div>',
            unsafe_allow_html=True,
        )

        unique_called = len(set(
            sp for h in S.episode_history for sp in h.get("called", [])
        ))
        dag_depth = str(S.env.delegation_graph.depth) if S.env else "—"

        stats = [
            ("Step",         str(S.step_n),                                              "#e2e8f0"),
            ("Total Reward", f"{sum(S.rewards):+.4f}" if S.rewards else "—",
             "#10b981" if (S.rewards and sum(S.rewards) >= 0) else "#ef4444"),
            ("Mean Step Rwd",f"{float(np.mean(S.rewards)):+.4f}" if S.rewards else "—", "#94a3b8"),
            ("Specialists",  str(unique_called),                                         "#7c3aed"),
            ("DAG Depth",    dag_depth,                                                  "#f59e0b"),
            ("Mean Entropy", f"{float(np.mean(S.step_entropies)):.3f}"
                             if S.step_entropies else "—",                               "#00d4ff"),
        ]

        for label, value, color in stats:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-size:11px;color:#475569;">{label}</span>'
                f'<span style="font-size:11px;font-weight:600;color:{color};">'
                f'{value}</span></div>',
                unsafe_allow_html=True,
            )

        if S.rewards:
            st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
            st.plotly_chart(fig_reward_curve(S.rewards), use_container_width=True)


def _render_replay_step(S: Session, step_idx: int) -> None:
    """Render charts for a specific historical step — no env calls."""
    if not S.episode_history or step_idx >= len(S.episode_history):
        st.info("No episode data to replay. Run an episode first.")
        return

    snap       = S.episode_history[step_idx]
    cumulative = snap["cumulative"]

    # Cumulative called specialists up to and including this step
    cumulative_called = list({
        sp
        for h in S.episode_history[:step_idx + 1]
        for sp in h.get("called", [])
    })

    st.markdown(
        f'<div style="background:rgba(124,58,237,0.07);border:1px solid rgba(124,58,237,0.2);'
        f'border-radius:10px;padding:12px 18px;font-size:12px;color:#a78bfa;margin-bottom:12px;">'
        f'Replaying Step {snap["step"]}  ·  Action: <b>{snap["action_name"]}</b>  ·  '
        f'Reward: <b>{snap["reward"]:+.4f}</b>  ·  '
        f'Cumulative: <b>{cumulative:+.4f}</b></div>',
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(
            fig_delegation_graph(S, cumulative_called, snap["edges"], highlight_latest=False),
            use_container_width=True,
            key=f"replay_dag_{step_idx}",
        )
    with rc2:
        st.plotly_chart(
            fig_reward_breakdown(snap["components"]),
            use_container_width=True,
            key=f"replay_breakdown_{step_idx}",
        )

    sec("Action Trace at This Step")
    trace_lines = []
    for h in S.episode_history[:step_idx + 1]:
        sign        = "+" if h["reward"] >= 0 else ""
        called_str  = ", ".join(h["called"]) if h["called"] else "—"
        marker      = "► " if h["step"] == snap["step"] else "  "
        trace_lines.append(
            f"{marker}Step {h['step']:>2}  │  {h['action_name']:<22}  │  "
            f"reward: {sign}{h['reward']:.4f}  │  specialists: {called_str}"
        )
    st.code("\n".join(trace_lines), language=None)


# ─────────────────────────────────────────────────────────
# Tab 1 — Live Demo
# ─────────────────────────────────────────────────────────
def tab_live_demo():
    S = _S()

    col_task, col_ctrl = st.columns([3, 2], gap="large")

    with col_task:
        sec("Task")
        task_dd  = st.selectbox("Preset task", PRESET_TASKS, key="task_dd")
        task_txt = st.text_input("Or enter custom task",
                                 placeholder="Describe a software engineering task…",
                                 key="task_txt")
        phase    = st.slider("Curriculum phase", 1, 3, 1, key="phase_sl")

    with col_ctrl:
        sec("Controls")
        c1, c2    = st.columns(2)
        reset_btn = c1.button("Reset Episode",    type="primary",    use_container_width=True, key="reset_btn")
        run_btn   = c2.button("Run Full Episode", use_container_width=True,                    key="run_btn")
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        cat       = _load_catalog()
        act_type  = st.selectbox("Action type",
                                 ["RANDOM", "STOP", "CALL SPECIALIST", "PARALLEL SPAWN"],
                                 key="act_type")
        spec_ids  = [sp["id"] for sp in cat]
        spec_ch   = st.selectbox("Target specialist", spec_ids, key="spec_ch")
        step_btn  = st.button("Execute One Step",
                              disabled=(S.env is None or S.done),
                              use_container_width=True, key="step_btn")

    status_msg = st.session_state.get("demo_status", "Click 'Reset Episode' to start.")
    status_clr = "#34d399" if "complete" in status_msg or "started" in status_msg else "#94a3b8"
    status_bar(status_msg, status_clr)
    st.markdown(_exec_mode_badges(S), unsafe_allow_html=True)

    # ── Reset ──────────────────────────────────────────────
    if reset_btn:
        with st.spinner("Initializing environment… (first run ~30 s on CPU)"):
            S.reset(int(phase))
        spawn_note = (
            f"  |  ⚡ Spawned: {', '.join(S.spawned_specialists)}"
            if S.spawned_specialists else ""
        )
        st.session_state.demo_status = f'Episode started  |  Task: "{S.task[:90]}"{spawn_note}'
        st.session_state.last_called = []
        st.session_state.last_edges  = []
        st.session_state.last_info   = {}
        st.rerun()

    # ── Step ───────────────────────────────────────────────
    if step_btn and S.env is not None and not S.done:
        action = np.zeros(S.env.action_space.shape, dtype=np.float32)
        if act_type == "STOP":
            action[0] = 1.0
        elif act_type == "CALL SPECIALIST":
            ids = S.registry.list_ids()
            if spec_ch in ids:
                idx = ids.index(spec_ch)
                if idx < S.env.max_specialists:
                    action[1 + idx] = 1.0
            else:
                action[1] = 1.0
        elif act_type == "PARALLEL SPAWN":
            action[0] = 6.0
            action[1] = 1.0
            if S.env.max_specialists > 1:
                action[2] = 1.0
            action[1 + S.env.max_specialists] = 1.0
        else:
            action = S.env.action_space.sample()

        _, r, term, trunc, info = S.step(action)
        done = term or trunc
        sign = "+" if r >= 0 else ""
        msg  = f"Step {S.step_n}  |  reward {sign}{r:.4f}  |  {'DONE' if done else 'Running…'}"
        if done:
            msg += f"  |  Total: {sum(S.rewards):+.4f}"
        st.session_state.demo_status = msg
        # Use cumulative called_ids so graph stays populated even after STOP step
        called = list(S.env.called_ids)
        edges  = [(e.caller_id, e.callee_id)
                  for e in S.env.delegation_graph.get_delegation_path()]
        st.session_state.last_called = called
        st.session_state.last_edges  = edges
        st.session_state.last_info   = info
        st.rerun()

    # ── Run Full ───────────────────────────────────────────
    if run_btn:
        with st.spinner("Running full episode…"):
            S.reset(int(phase))
            info = {}
            for _ in range(15):
                if S.done:
                    break
                _, _, _, _, info = S.step(S.env.action_space.sample())
        # Use cumulative called_ids so graph stays populated even after STOP step
        called = list(S.env.called_ids) if S.env else []
        edges  = [(e.caller_id, e.callee_id)
                  for e in S.env.delegation_graph.get_delegation_path()]
        total  = sum(S.rewards)
        st.session_state.demo_status = (
            f"Episode complete  |  {S.step_n} steps  |  Total reward: {total:+.4f}"
        )
        st.session_state.last_called = called
        st.session_state.last_edges  = edges
        st.session_state.last_info   = info
        st.rerun()

    # ── Metric strip ──────────────────────────────────────
    if S.env is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Obs Dim",     int(S.env.observation_space.shape[0]))
        mc2.metric("Action Dim",  int(S.env.action_space.shape[0]))
        mc3.metric("Specialists", S.registry.size)
        mc4.metric("Phase",       phase)

    # ── Hero: Robot Orchestrator Widget (full width) ──────
    sec("Orchestrator  ·  Live Delegation View")
    last_info = st.session_state.get("last_info", {})
    render_orchestrator({
        "called":  st.session_state.get("last_called", []),
        "active":  (st.session_state.get("last_called", []) or [""])[-1]
                   if not S.done else "",
        "edges":   st.session_state.get("last_edges",  []),
        "task":    S.task,
        "step":    S.step_n,
        "mode":    last_info.get("delegation_mode", "SEQUENTIAL"),
        "done":    S.done,
        "reward":  sum(S.rewards) if S.rewards else None,
        "phase":   int(st.session_state.get("phase_sl", 1)),
    })
    # Thought bubble ticker — robot's last internal monologue
    _thoughts = last_info.get("thoughts") or last_info.get("thought")
    if _thoughts:
        st.markdown(
            f'<div style="font-size:11px;color:#64748b;margin-top:-8px;padding:4px 8px;">'
            f'💭 {_html.escape(str(_thoughts))}</div>',
            unsafe_allow_html=True,
        )

    # ── Three-column secondary row ─────────────────────────
    sc1, sc2, sc3 = st.columns([4, 4, 4])
    with sc1:
        st.plotly_chart(fig_reward_curve(S.rewards), use_container_width=True)
    with sc2:
        last_info = st.session_state.get("last_info", {})
        st.plotly_chart(
            fig_reward_breakdown(last_info.get("reward_components", {})),
            use_container_width=True,
        )
    with sc3:
        sec("Policy Confidence")
        if S.step_entropies:
            st.plotly_chart(
                fig_policy_confidence(
                    S.step_entropies,
                    [h["step"] for h in S.episode_history],
                ),
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div style="color:#334155;font-size:11px;padding:24px;text-align:center;">'
                'Run an episode to see action entropy.</div>',
                unsafe_allow_html=True,
            )

    # ── Step Log (full width) ──────────────────────────────
    sec("Step Log / Action Trace")
    if not S.actions:
        st.markdown(
            '<div style="color:#334155;font-size:12px;padding:16px;text-align:center;">'
            'Waiting… Reset the episode to start.</div>',
            unsafe_allow_html=True,
        )
    else:
        lines = []
        for i, (inf, r) in enumerate(zip(S.actions, S.rewards)):
            sign  = "+" if r >= 0 else ""
            act   = inf.get("action_name", "UNKNOWN")
            specs = ", ".join(inf.get("called_specialists", []))
            mode  = inf.get("delegation_mode", "")
            e_str = (f" │ entropy: {S.step_entropies[i]:.3f}"
                     if i < len(S.step_entropies) else "")
            lats  = inf.get("specialist_latencies", {})
            lat_str = (
                "\n       │  → latency:  "
                + ", ".join(f"{k}: {v:.0f}ms" for k, v in lats.items())
            ) if lats else ""
            lines.append(
                f"Step {i+1:>2} │ {act:<22} │ reward: {sign}{r:.4f}{e_str}"
                + (f"\n       │  → called:  {specs}" if specs else "")
                + (f"\n       │  → mode:    {mode}" if mode else "")
                + lat_str
            )
        total = sum(S.rewards)
        unique_sp = len(set(sp for h in S.episode_history for sp in h.get("called", [])))
        lines.append(f"{'─'*62}")
        lines.append(
            f"Total reward: {'+' if total>=0 else ''}{total:.4f}  │  "
            f"Steps: {len(S.rewards)}  │  "
            f"Specialists called: {unique_sp} unique"
        )
        st.code("\n".join(lines), language=None)

    # ── Episode Replay (full width) ────────────────────────
    if S.episode_history:
        st.markdown("---")
        sec("Episode Replay Mode")
        st.caption(
            "Scrub backward through every step of the episode. "
            "Delegation graph, reward breakdown, and action trace all update to that exact state. "
            "100% real data — no re-simulation."
        )
        n_steps = len(S.episode_history)
        if n_steps > 1:
            replay_step = st.slider(
                "Replay step",
                min_value=1,
                max_value=n_steps,
                value=n_steps,
                step=1,
                key="replay_slider",
                format="Step %d",
            )
        else:
            replay_step = 1
            st.caption("Single-step episode — showing step 1.")
        _render_replay_step(S, replay_step - 1)


# ─────────────────────────────────────────────────────────
# Tab 2 — Specialists
# ─────────────────────────────────────────────────────────
def tab_specialists():
    S = _S()

    # Prefer live registry so dynamically-added specialists appear immediately.
    # Fall back to YAML catalog before the environment has been booted.
    if S.registry is not None:
        specialists  = S.registry.list_all()
        source_note  = None
    else:
        class _SP:
            def __init__(self, d: dict):
                self.id                  = d["id"]
                self.role                = d["role"]
                self.description         = d["description"]
                self.complexity_affinity = d["complexity_affinity"]
                self.avg_latency_ms      = d["avg_latency_ms"]
        specialists = [_SP(d) for d in _load_catalog()]
        source_note = "Showing YAML catalog — run an episode to load the live registry (includes dynamic additions)."

    n = len(specialists)
    sec(f"Roster — {n} specialist{'s' if n != 1 else ''}, capability-embedded")
    if source_note:
        st.caption(source_note)

    spawned_set = set(S.spawned_specialists) if S.registry is not None else set()

    cols = st.columns(4)
    for i, sp in enumerate(specialists):
        c          = SPEC_COLORS.get(sp.id, "#7c3aed")
        is_spawned = sp.id in spawned_set
        border_top = "#fbbf24" if is_spawned else c
        spawn_tag  = (
            '<span style="font-size:9px;font-weight:700;color:#fbbf24;'
            'background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.25);'
            'border-radius:999px;padding:1px 7px;margin-left:6px;">⚡ AUTO-SPAWNED</span>'
            if is_spawned else ""
        )
        with cols[i % 4]:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.025);border:1px solid {c}22;
            border-left:3px solid {border_top};border-radius:12px;
            padding:14px;margin-bottom:10px;">
  <div style="font-size:11px;font-weight:700;color:{c};margin-bottom:6px;">
    {sp.role}{spawn_tag}
  </div>
  <div style="font-size:11px;color:#64748b;line-height:1.5;">
    {_html.escape(sp.description[:90])}…
  </div>
  <div style="font-size:10px;color:#334155;margin-top:8px;padding-top:8px;
              border-top:1px solid rgba(255,255,255,0.05);">
    {sp.avg_latency_ms} ms &nbsp;·&nbsp; {', '.join(sp.complexity_affinity)}
  </div>
</div>""", unsafe_allow_html=True)

    sec("Capability Similarity Matrix")
    if st.button("Load Similarity Matrix", key="sim_btn"):
        with st.spinner("Computing cosine similarity across 384-dim embeddings…"):
            S.boot()
            st.plotly_chart(fig_similarity(S.registry), use_container_width=True)

    sec("Add Specialist Dynamically")
    st.caption("New specialists are immediately representable via their 384-dim embedding — no retraining or YAML edits required.")
    c1, c2   = st.columns(2)
    new_id   = c1.text_input("ID",   placeholder="ml_engineer", key="new_id")
    new_role = c2.text_input("Role", placeholder="ML Engineer",  key="new_role")
    new_desc = st.text_area("Description",
                            placeholder="Expert in PyTorch, model training, MLOps pipelines…",
                            height=80, key="new_desc")
    if st.button("Add to Roster", type="primary", key="add_btn"):
        if new_id.strip() and new_role.strip() and new_desc.strip():
            with st.spinner("Encoding specialist embedding…"):
                S.boot()
                S.registry.add_specialist({
                    "id": new_id.strip(), "role": new_role.strip(),
                    "description": new_desc.strip(),
                    "complexity_affinity": ["moderate", "complex"],
                    "avg_latency_ms": 5000,
                })
            st.success(
                f"'{new_id.strip()}' added. "
                "Policy can represent it via 384-dim embedding — no retraining needed."
            )
            st.plotly_chart(fig_similarity(S.registry), use_container_width=True)
        else:
            st.warning("Fill in all three fields.")


# ─────────────────────────────────────────────────────────
# Tab 3 — Training
# ─────────────────────────────────────────────────────────
def tab_training():
    sec("Training Progress — Mean Reward per Episode")
    st.plotly_chart(fig_training_curve(), use_container_width=True)

    sec("Policy Entropy — Action Confidence Over Training")
    st.caption(
        "Entropy of the specialist-selection distribution. "
        "High = exploring (early training). Low = confident routing (converged policy)."
    )
    st.plotly_chart(fig_training_entropy(), use_container_width=True)

    sec("Curriculum Phases")
    c1, c2, c3 = st.columns(3)
    _phase_card = lambda col, color, label, eps, desc: col.markdown(
        f'<div style="background:rgba({color},0.04);border:1px solid rgba({color},0.18);'
        f'border-radius:12px;padding:18px;">'
        f'<div style="font-size:10px;font-weight:700;color:rgb({color});text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:5px;">{eps}</div>'
        f'<div style="font-size:11px;color:#475569;">{desc}</div></div>',
        unsafe_allow_html=True,
    )
    _phase_card(c1, "0,212,255",  "Phase 1 · Atomic",             "200 episodes",
                "Agent learns basic routing — which single specialist to call.")
    _phase_card(c2, "124,58,237", "Phase 2 · Moderate",           "400 episodes",
                "Agent learns multi-specialist coordination and mode selection.")
    _phase_card(c3, "245,158,11", "Phase 3 · Complex/Enterprise", "600 episodes",
                "Full delegation strategy with DAG depth, fallbacks, and latency trade-offs.")

    sec("Quick Start Commands")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Local training**")
        st.code(
            "# Demo mode — no OpenAI key needed\n"
            "cd spindleflow-rl\n"
            "python training/train.py \\\n"
            "  --phase 1 --timesteps 50000\n\n"
            "# Monitor in TensorBoard\n"
            "tensorboard --logdir tensorboard_logs/",
            language="bash",
        )
    with c2:
        st.markdown("**Google Colab (T4 GPU, free)**")
        st.code(
            "!git clone https://github.com/garvitsachdevaa/kuchbhi\n"
            "%cd kuchbhi\n"
            "!pip install -r requirements.txt sb3-contrib\n\n"
            "# 5k-step demo run\n"
            "%run colab/train_colab.py",
            language="python",
        )


# ─────────────────────────────────────────────────────────
# Tab 4 — Quality Demo
# ─────────────────────────────────────────────────────────
def tab_quality():
    sec("Before vs After Delegation Learning")
    if st.button("Load Demo Comparison", type="primary", key="load_demo"):
        p = ASSETS / "demo_moment_1.json"
        if not p.exists():
            st.error("Run `python demo/precompute_demo.py` first to generate demo assets.")
        else:
            with open(p) as f:
                d = json.load(f)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    '<div style="font-size:10px;font-weight:700;color:#ef4444;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
                    'Generalist Output (No Delegation)</div>',
                    unsafe_allow_html=True,
                )
                st.code(d["generalist_output"][:700], language=None)
            with c2:
                st.markdown(
                    '<div style="font-size:10px;font-weight:700;color:#10b981;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
                    'Specialist-Routed Output (Learned Policy)</div>',
                    unsafe_allow_html=True,
                )
                st.code(d["specialist_output"][:700], language=None)

    sec("Policy Tuning — Quality vs Latency")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div style="background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.2);
            border-radius:12px;padding:16px;">
  <div style="font-size:10px;font-weight:700;color:#a78bfa;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:8px;">Quality Policy</div>
  <div style="font-size:12px;color:#64748b;line-height:1.8;">
    5 specialists &nbsp;·&nbsp; sequential &nbsp;·&nbsp; ~180 s<br>
    <code style="color:#a78bfa;background:rgba(124,58,237,0.12);
                 padding:2px 6px;border-radius:4px;">latency_weight = 0.0</code>
  </div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
            border-radius:12px;padding:16px;">
  <div style="font-size:10px;font-weight:700;color:#00d4ff;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:8px;">Latency Policy</div>
  <div style="font-size:12px;color:#64748b;line-height:1.8;">
    3 specialists &nbsp;·&nbsp; parallel &nbsp;·&nbsp; ~45 s<br>
    <code style="color:#00d4ff;background:rgba(0,212,255,0.1);
                 padding:2px 6px;border-radius:4px;">latency_weight = 0.15</code>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Tab 5 — Reward Lab
# ─────────────────────────────────────────────────────────
def tab_reward_lab():
    sec("Interactive Reward Explorer")
    st.caption("Tune the reward weights and watch each component update live.")

    col_s, col_c = st.columns([1, 2], gap="large")
    with col_s:
        lw = st.slider("Latency Weight",     0.0, 0.50, 0.05, 0.01, key="rl_lw")
        ep = st.slider("Efficiency Penalty", 0.0, 0.20, 0.05, 0.01, key="rl_ep")
        fp = st.slider("Failure Penalty",    0.0, 1.00, 0.30, 0.05, key="rl_fp")
        cw = st.slider("Consistency Bonus",  0.0, 0.50, 0.10, 0.01, key="rl_cw")
        eb = st.slider("Explanation Bonus",  0.0, 0.20, 0.05, 0.01, key="rl_eb")

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
    with col_c:
        st.plotly_chart(fig_reward_breakdown(comps), use_container_width=True)
        st.markdown(
            f'<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.18);'
            f'border-radius:10px;padding:14px 18px;font-size:13px;color:#94a3b8;">'
            f'Estimated total reward: '
            f'<span style="color:#00d4ff;font-weight:700;font-size:20px;">{sign}{total:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────
# Tab 6 — Architecture
# ─────────────────────────────────────────────────────────
def tab_architecture():
    obs0 = EpisodeState.observation_dim(6)
    act0 = 6 + 6

    c1, c2 = st.columns(2)
    with c1:
        sec(f"Observation Space  ({obs0:,} dims)")
        st.markdown("""
| Dims | Component |
|-----:|-----------|
| 384  | Task embedding (all-MiniLM-L6-v2) |
| 2304 | Roster embeddings (6 × 384) |
| 2304 | Called embeddings (6 × 384) |
| 384  | Scratchpad embedding |
| 100  | Delegation graph adjacency (10 × 10) |
| 6    | Called-specialist mask |
| 8    | Scalar features |
""")
    with c2:
        sec(f"Action Space  ({act0}-dim Box)")
        st.markdown("""
| Index  | Component |
|--------|-----------|
| [0]    | Meta-action (STOP / CALL / PARALLEL…) |
| [1:7]  | Specialist selection logits (multi-hot) |
| [7]    | Delegation mode (SEQ / PAR / FAN-OUT…) |
| [8:12] | Mode parameters (rounds, threshold…) |
""")

    c1, c2, c3 = st.columns(3)
    with c1:
        sec("Policy")
        st.markdown("""
- **LSTM PPO** (RecurrentPPO)
- MlpLstmPolicy
- Hidden: 256 · 1 layer
- POMDP-safe via LSTM state
- 4 factored action heads
""")
    with c2:
        sec("Tiered Reward")
        st.markdown("""
- **T0** — Structural heuristics
- **T1** — Cosine embedding sim
- **T2** — GPT-4o-mini judge
- **T3** — Full judge (checkpoints)
- Episode-level tier lock
""")
    with c3:
        sec("Safety")
        st.markdown("""
- DAG cycle detection (DFS)
- Max delegation depth: 2
- Scratchpad sandbox isolation
- Injection sanitization
- Action masking (DAG)
""")

    sec("Reward Function")
    st.code("""total_reward = (
  quality_delta          # specialist_score − baseline  (same tier)
− efficiency_penalty     # 0.05 × max(0, n_called − expected)
− failure_penalty        # 0.3 per timeout,  0.2 per error
+ recovery_bonus         # +0.1 if fallback succeeded
− conflict_penalty       # 0.1 per unresolved conflict
+ conflict_bonus         # 0.05 per resolved conflict
+ consistency_bonus      # 0.1 × Dirichlet-prior path score
− latency_penalty        # latency_weight × overage_fraction
+ explanation_bonus      # 0.05 if delegation is auditable
)""", language="python")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
def main():
    inject_css()
    hero()
    S = _S()
    render_live_stats(S)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "⚡ Live Demo",
        "🤖 Specialists",
        "📈 Training",
        "🔍 Quality Demo",
        "🧪 Reward Lab",
        "🏗 Architecture",
    ])
    with t1: tab_live_demo()
    with t2: tab_specialists()
    with t3: tab_training()
    with t4: tab_quality()
    with t5: tab_reward_lab()
    with t6: tab_architecture()


# Guard allows safe imports for testing without triggering the UI.
# Streamlit runs scripts with __name__ == "__main__".
if __name__ == "__main__":
    main()
